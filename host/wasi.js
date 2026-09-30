// The WASI a browser gives a snaidhm program: read-only files it was handed,
// its arguments, stdout and stderr.
//
// Almide's wasm target reaches the outside world through WASI, and a font is
// just a file to it — `fs.read_bytes_raw(path)` does the same thing natively
// and here. So the page fetches the fonts and passes them in as files; there is
// no font-specific import namespace.
//
// What is implemented is what a program that reads files uses: one preopened
// directory `/` holding `files`, path_open / fd_read / fd_filestat_get /
// fd_close on them, args, an empty environment, and fd_write to stdout and
// stderr. Anything else answers ENOSYS (52) instead of pretending to succeed.
//
//   const w = createWasi({ files: { "/fonts/ui.ttf": bytes }, args: ["app"] });
//   const { instance } = await WebAssembly.instantiate(wasm, { wasi_snapshot_preview1: w.imports, ... });
//   w.setMemory(instance.exports.memory);

const ESUCCESS = 0, EBADF = 8, EINVAL = 28, ENOENT = 44, ENOSYS = 52;
const PREOPEN = 3;
const FILETYPE_DIRECTORY = 3, FILETYPE_REGULAR = 4;

/// files: path (absolute, "/"-separated) -> Uint8Array | ArrayBuffer.
export function createWasi({ files = {}, args = [], stdout, stderr } = {}) {
  let memory = null;
  const u8 = () => new Uint8Array(memory.buffer);
  const view = () => new DataView(memory.buffer);
  const utf8 = new TextEncoder(), utf8d = new TextDecoder();

  const table = new Map();
  for (const [path, data] of Object.entries(files)) {
    table.set(path.replace(/^\/+/, ""), data instanceof Uint8Array ? data : new Uint8Array(data));
  }
  // fd -> { data, pos } for opened files; fds after the preopen.
  const open = new Map();
  let nextFd = PREOPEN + 1;

  const argBytes = args.map((a) => utf8.encode(a + "\0"));
  const lines = { 1: "", 2: "" };
  const sinks = {
    1: stdout ?? ((s) => console.log(s)),
    2: stderr ?? ((s) => console.error(s)),
  };

  const imports = {
    args_sizes_get(countPtr, sizePtr) {
      view().setUint32(countPtr, argBytes.length, true);
      view().setUint32(sizePtr, argBytes.reduce((n, a) => n + a.length, 0), true);
      return ESUCCESS;
    },
    args_get(argvPtr, bufPtr) {
      let p = bufPtr;
      argBytes.forEach((a, i) => {
        view().setUint32(argvPtr + i * 4, p, true);
        u8().set(a, p);
        p += a.length;
      });
      return ESUCCESS;
    },
    environ_sizes_get(countPtr, sizePtr) {
      view().setUint32(countPtr, 0, true);
      view().setUint32(sizePtr, 0, true);
      return ESUCCESS;
    },
    environ_get() { return ESUCCESS; },

    fd_prestat_get(fd, ptr) {
      if (fd !== PREOPEN) return EBADF;
      view().setUint8(ptr, 0); // a directory
      view().setUint32(ptr + 4, 1, true); // its name, "/"
      return ESUCCESS;
    },
    fd_prestat_dir_name(fd, ptr, len) {
      if (fd !== PREOPEN) return EBADF;
      if (len < 1) return EINVAL;
      u8()[ptr] = 0x2f;
      return ESUCCESS;
    },
    path_open(dirfd, _lookup, pathPtr, pathLen, _oflags, _rights, _inherit, _fdflags, fdPtr) {
      if (dirfd !== PREOPEN) return EBADF;
      const path = utf8d.decode(u8().slice(pathPtr, pathPtr + pathLen)).replace(/^\/+/, "");
      const data = table.get(path);
      if (!data) return ENOENT;
      const fd = nextFd++;
      open.set(fd, { data, pos: 0 });
      view().setUint32(fdPtr, fd, true);
      return ESUCCESS;
    },
    fd_filestat_get(fd, ptr) {
      const f = open.get(fd);
      if (!f && fd !== PREOPEN) return EBADF;
      const v = view();
      for (let i = 0; i < 64; i += 8) v.setBigUint64(ptr + i, 0n, true);
      v.setUint8(ptr + 16, f ? FILETYPE_REGULAR : FILETYPE_DIRECTORY);
      v.setBigUint64(ptr + 24, 1n, true); // nlink
      v.setBigUint64(ptr + 32, BigInt(f ? f.data.length : 0), true);
      return ESUCCESS;
    },
    fd_read(fd, iovs, iovsLen, nreadPtr) {
      const f = open.get(fd);
      if (!f) return EBADF;
      let total = 0;
      for (let i = 0; i < iovsLen; i++) {
        const buf = view().getUint32(iovs + i * 8, true);
        const len = view().getUint32(iovs + i * 8 + 4, true);
        const chunk = f.data.subarray(f.pos, f.pos + len);
        u8().set(chunk, buf);
        f.pos += chunk.length;
        total += chunk.length;
        if (chunk.length < len) break;
      }
      view().setUint32(nreadPtr, total, true);
      return ESUCCESS;
    },
    fd_close(fd) {
      return open.delete(fd) ? ESUCCESS : EBADF;
    },
    fd_write(fd, iovs, iovsLen, nwrittenPtr) {
      if (fd !== 1 && fd !== 2) return EBADF;
      let total = 0;
      for (let i = 0; i < iovsLen; i++) {
        const buf = view().getUint32(iovs + i * 8, true);
        const len = view().getUint32(iovs + i * 8 + 4, true);
        lines[fd] += utf8d.decode(u8().slice(buf, buf + len), { stream: true });
        total += len;
      }
      // Hand over whole lines; a partial one waits for its newline.
      const cut = lines[fd].lastIndexOf("\n");
      if (cut >= 0) {
        sinks[fd](lines[fd].slice(0, cut));
        lines[fd] = lines[fd].slice(cut + 1);
      }
      view().setUint32(nwrittenPtr, total, true);
      return ESUCCESS;
    },
    proc_exit(code) {
      throw new WasiExit(code);
    },
  };

  return {
    imports: new Proxy(imports, {
      get: (t, name) => t[name] ?? (() => ENOSYS),
    }),
    setMemory(m) { memory = m; },
    /// Hand over what is left of stdout and stderr without a newline.
    flush() {
      for (const fd of [1, 2]) if (lines[fd]) { sinks[fd](lines[fd]); lines[fd] = ""; }
    },
  };
}

/// Thrown by proc_exit so the caller of `_start` learns the exit code.
export class WasiExit extends Error {
  constructor(code) { super(`exit ${code}`); this.code = code; }
}
