# LumaViewPro — Third-Party Notices

LumaViewPro is distributed under the MIT License (see `docs/LICENSE`). It bundles and uses software from the third parties listed below. Each entry identifies the component, the license under which it is distributed, and the upstream source. Full license texts for each license family are provided in `docs/licenses/`.

This NOTICE file is maintained by Etaluma, Inc. in accordance with Apache-2.0 §4(d) and comparable attribution requirements from BSD, LGPL, and GPL-with-Classpath-Exception licenses that apply to bundled components.

**Last updated**: 2026-10-29 (for LumaViewPro 4.0.0-beta). Every package `requirements.txt` names is listed at its pin, its license read from its own Windows wheel. The packages those pull in (FastAPI's, uvicorn's and the rest) are still to be listed from a Windows build environment.
**Regenerating**: For each package `requirements.txt` names, read the license its Windows wheel declares (`pip download --no-deps --platform win_amd64 --only-binary=:all: <name>==<pin>`, then the wheel's `METADATA` and its license files); for the packages they pull in, the build environment's own `pip freeze`.

---

## Bundled Python packages (pip-installed and shipped inside the application folder)

| Package | Version | License | Upstream |
|---|---|---|---|
| Kivy | 2.3.1 | MIT | https://kivy.org/ |
| kivy-deps (sdl2, glew) | bundled with Kivy | zlib / MIT (SDL2 family) | https://www.libsdl.org/ |
| NumPy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org/ |
| SciPy | 1.17.1 | BSD-3-Clause | https://scipy.org/ |
| Matplotlib | 3.10.9 | Matplotlib License (PSF-based, BSD-compatible) | https://matplotlib.org/ |
| pandas | 3.0.3 | BSD-3-Clause | https://pandas.pydata.org/ |
| scikit-image | 0.26.0 | BSD-3-Clause | https://scikit-image.org/ |
| opencv-python-headless | 4.13.0.92 | Apache-2.0 | https://opencv.org/ |
| tifffile | 2026.5.15 | BSD-3-Clause | https://github.com/cgohlke/tifffile |
| imagecodecs | 2026.5.10 | BSD-3-Clause (wrapper) | https://github.com/cgohlke/imagecodecs |
| pyserial | 3.5 | BSD-3-Clause | https://github.com/pyserial/pyserial |
| psutil | 7.2.2 | BSD-3-Clause | https://github.com/giampaolo/psutil |
| platformdirs | 4.9.6 | MIT | https://github.com/platformdirs/platformdirs |
| pypylon | 26.4.1 | BSD-3-Clause (the wrapper; the wheel's metadata declares Other/Proprietary). It carries the Basler pylon runtime, below. | https://github.com/basler/pypylon |
| ids-peak | 1.15.0.0.8 | IDS license terms (proprietary) | https://pypi.org/project/ids-peak/ |
| ids-peak-ipl | 1.17.2.0.7 | IDS license terms (proprietary) | https://pypi.org/project/ids-peak-ipl/ |
| ids-peak-icv | 1.3.0.9744 | IDS license terms (proprietary) | https://pypi.org/project/ids-peak-icv/ |
| ids-peak-afl | 2.0.1.0.6 | IDS license terms (proprietary) | https://pypi.org/project/ids-peak-afl/ |
| PyAV (`av`) | 17.0.1 | BSD-3-Clause wrapper over LGPL-2.1 FFmpeg | https://github.com/PyAV-Org/PyAV |
| FastAPI | 0.143.0 | MIT | https://github.com/fastapi/fastapi |
| uvicorn | 0.54.0 | BSD-3-Clause | https://github.com/Kludex/uvicorn |
| pydantic | 2.14.0 | MIT | https://github.com/pydantic/pydantic |
| Starlette | 1.7.0 | BSD-3-Clause | https://github.com/Kludex/starlette |
| pyusb | 1.3.1 | BSD-3-Clause | https://github.com/pyusb/pyusb |
| libusb1 (python-libusb1) | 3.4.0 | LGPL-2.1-or-later | https://github.com/vpelletier/python-libusb1 |
| libusb-package | 1.0.30.0 | Apache-2.0 (packaging; carries the LGPL-2.1 libusb library below) | https://github.com/pyocd/libusb-package |

The IDS license terms permit distributing the IDS software only in products that operate with IDS cameras. Each ids-peak wheel carries those terms (`LICENSE`) and the licenses of the open-source components inside it (`third_party_licenses.txt`, `third_party_copyrights.txt`; ids-peak's GenICam licenses under `genicam/`).

## C / C++ libraries bundled inside the imagecodecs wheel

These libraries are statically or dynamically linked into `imagecodecs` and are shipped inside the application. All are distributed under permissive licenses; see `docs/licenses/` for the license text families.

| Library | License | Upstream |
|---|---|---|
| libjpeg-turbo / libjpeg | BSD-3-Clause / IJG | https://libjpeg-turbo.org/ |
| libtiff | libtiff (BSD-like) | http://www.libtiff.org/ |
| libpng | PNG Reference Library License v2 (BSD-like) | http://www.libpng.org/pub/png/libpng.html |
| zlib | zlib License (BSD-like) | https://www.zlib.net/ |
| libwebp | BSD-3-Clause | https://developers.google.com/speed/webp |
| OpenJPEG | BSD-2-Clause | https://www.openjpeg.org/ |
| Zstandard (zstd) | BSD-3-Clause | https://github.com/facebook/zstd |
| XZ Utils / liblzma | 0BSD / Public Domain | https://tukaani.org/xz/ |
| c-blosc / c-blosc2 | BSD-3-Clause | https://github.com/Blosc/c-blosc |
| Snappy | BSD-3-Clause | https://github.com/google/snappy |
| Brotli | MIT | https://github.com/google/brotli |
| JPEG XL / libjxl (if enabled) | BSD-3-Clause | https://github.com/libjxl/libjxl |

**Authoritative per-build list**: the imagecodecs wheel carries each codec's license text in its `imagecodecs/licenses/` folder. The installer does not yet carry that folder; copying it in is a build step still to come.

## Bundled binaries (not pip-installed)

These ship inside the application folder after PyInstaller builds the Windows installer:

| Binary | License | Upstream | How it's bundled |
|---|---|---|---|
| FFmpeg shared libraries (`avcodec`, `avformat`, `avutil`, `swscale`, `swresample`) | LGPL-2.1-or-later | https://ffmpeg.org/download.html | Dynamically linked from PyAV. LumaViewPro ships PyAV's stock PyPI wheels, whose FFmpeg binaries are built LGPL-only (no `--enable-gpl` components) per PyAV's upstream wheel build configuration. DLLs live in the application's `av.libs/` folder (or equivalent) and are user-replaceable per LGPL §6. |
| SDL2 + SDL2_image / SDL2_mixer / SDL2_ttf | zlib license | https://www.libsdl.org/ | Bundled via `kivy_deps.sdl2` and included in the PyInstaller `.spec` via `Tree(sdl2.dep_bins)`. See `docs/licenses/LICENSE.*.txt` for each sub-component. |
| GLEW | Modified BSD / MIT | http://glew.sourceforge.net/ | Bundled via `kivy_deps.glew`. |
| OpenBLAS | BSD-3-Clause | https://www.openblas.net/ | Bundled inside NumPy/SciPy wheels. |
| Basler pylon runtime (`PylonBase`, `PylonUtility`, the USB and GigE transport layers, the GenICam `GenApi` / `GCBase` libraries) | Proprietary, Basler's: the wheel's metadata declares Other/Proprietary and it carries no license text for them | https://www.baslerweb.com/pylon | Shipped inside the pypylon wheel and collected with it. |
| IDS peak libraries (`ids_peak.dll`, `ids_peak_ipl.dll`, `ids_peak_icv.dll`, `ids_peak_afl.dll`, their GenICam libraries and TBB) | IDS license terms (proprietary); the open-source components' licenses as each wheel lists them | https://en.ids-imaging.com/ids-peak.html | Shipped inside the four ids-peak wheels, each collected whole. |
| libusb-1.0 (`libusb-1.0.dll` / `.dylib` / `.so`) | LGPL-2.1-or-later | https://libusb.info/ | The USB library the Lumascope Classic (LS560, LS620, LS720) drivers run on, through pyusb and python-libusb1. Shipped unmodified from the `libusb-package` wheel, in the application's `libusb_package/` folder, and loaded at run time, so it is user-replaceable per LGPL §6. |

## Installed by their own installers

The camera vendors' USB drivers are installed by the vendors' own installers, each under its own EULA. The runtime libraries LumaViewPro calls are not installed this way: they ship inside the pypylon and ids-peak wheels, above.

| SDK | License | Upstream |
|---|---|---|
| Basler Pylon SDK | Proprietary (Basler EULA) | https://docs.baslerweb.com/pylon-software-suite |
| IDS Peak SDK | Proprietary (IDS EULA) | https://en.ids-imaging.com/ids-peak.html |

The Windows Bundle installer *chains* Basler's signed pylon USB camera driver MSI, and, when the build has it, the IDS peak runtime installer, run silently.

---

## LumaViewPro copyright notice

LumaViewPro is Copyright (c) 2023-2026 Etaluma, Inc. and is distributed under the MIT License; see `docs/LICENSE`.

---

## License text files

Full license text for each distinct license used above is provided alongside this file:

- `docs/licenses/LICENSE.MIT.txt`
- `docs/licenses/LICENSE.BSD-2-Clause.txt`
- `docs/licenses/LICENSE.BSD-3-Clause.txt`
- `docs/licenses/LICENSE.0BSD.txt`
- `docs/licenses/LICENSE.Apache-2.0.txt`
- `docs/licenses/LICENSE.matplotlib.txt`
- `docs/licenses/LICENSE.LGPL-2.1.txt`
- Not yet in `docs/licenses/`: the CC0-1.0 text NumPy names, the IDS license terms and the licenses the ids-peak and imagecodecs wheels list; carrying them is a build step still to come.
- `docs/licenses/LICENSE.FLAC.txt`, `LICENSE.freetype.txt`, `LICENSE.harfbuzz.txt`, `LICENSE.jpeg.txt`, `LICENSE.modplug.txt`, `LICENSE.mpg123.txt`, `LICENSE.ogg-vorbis.txt`, `LICENSE.opus.txt`, `LICENSE.opusfile.txt`, `LICENSE.png.txt`, `LICENSE.tiff.txt`, `LICENSE.webp.txt`, `LICENSE.zlib.txt` — SDL2 sub-component licenses, pre-existing.
