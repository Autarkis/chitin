# Third-party notices

`coacd.wasm` in this package is compiled from third-party C++ sources. Their
license obligations are reproduced or referenced below. The `chitin` packaging
and JavaScript glue itself is MIT-licensed (see `LICENSE`).

## CoACD — MIT License

Source: <https://github.com/SarahWeiii/CoACD> (tag `1.0.11`)

```
MIT License

Copyright (c) 2022 Xinyue Wei, Minghua Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## CDT (Constrained Delaunay Triangulation) — Mozilla Public License 2.0

Source: <https://github.com/artem-ogre/CDT> (commit
`ec03b309fd18102ab1da069f2edf3b37be5d1fb3`, the revision pinned by CoACD 1.0.11)

CoACD statically links CDT, which is licensed under the **Mozilla Public License,
version 2.0**. The full license text is included in this package as
`LICENSE-MPL-2.0.txt`.

Under the MPL, the exact source of the MPL-covered portion (CDT) is available at
the URL and commit above. No modifications were made to CDT; it is compiled as a
dependency of CoACD.
