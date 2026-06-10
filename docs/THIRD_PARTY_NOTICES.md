# LumaViewPro — Third-Party Notices

LumaViewPro is distributed under the MIT License (see `docs/LICENSE`). It bundles and uses software from the third parties listed below. Each entry identifies the component, the license under which it is distributed, and the upstream source. Full license texts for each license family are provided in `docs/licenses/`.

This NOTICE file is maintained by Etaluma, Inc. in accordance with Apache-2.0 §4(d) and comparable attribution requirements from BSD, LGPL, and GPL-with-Classpath-Exception licenses that apply to bundled components.

**Last updated**: 2026-04-17 (for LumaViewPro 4.0.0-beta)
**Regenerating**: If dependencies materially change, re-run `pip-licenses --with-license-file --with-notice-file --format=markdown` on the build venv after `pip install -r requirements.txt` and reconcile against this file. For 4.0.0-beta, this file was authored from the declared `requirements.txt` + PyInstaller spec and each declared dep's documented upstream license.

---

## Bundled Python packages (pip-installed and shipped inside the application folder)

| Package | Version | License | Upstream |
|---|---|---|---|
| Kivy | 2.3.1 | MIT | https://kivy.org/ |
| kivy-deps (sdl2, glew) | bundled with Kivy | zlib / MIT (SDL2 family) | https://www.libsdl.org/ |
| NumPy | 2.4.4 | BSD-3-Clause | https://numpy.org/ |
| SciPy | 1.17.1 | BSD-3-Clause | https://scipy.org/ |
| Matplotlib | 3.10.8 | Matplotlib License (PSF-based, BSD-compatible) | https://matplotlib.org/ |
| pandas | 3.0.2 | BSD-3-Clause | https://pandas.pydata.org/ |
| scikit-image | 0.26.0 | BSD-3-Clause | https://scikit-image.org/ |
| opencv-python-headless | 4.13.0.92 | Apache-2.0 | https://opencv.org/ |
| tifffile | 2026.3.3 | BSD-3-Clause | https://github.com/cgohlke/tifffile |
| imagecodecs | 2026.3.6 | BSD-3-Clause (wrapper) | https://github.com/cgohlke/imagecodecs |
| pyserial | 3.5 | BSD-3-Clause | https://github.com/pyserial/pyserial |
| psutil | 7.2.2 | BSD-3-Clause | https://github.com/giampaolo/psutil |
| platformdirs | 4.4.0 | MIT | https://github.com/platformdirs/platformdirs |
| pypylon | 26.4.1 | BSD-3-Clause (Python wrapper only) | https://github.com/basler/pypylon |
| ids-peak | 1.13.0.0.6 | BSD-3-Clause (Python wrapper only) | https://pypi.org/project/ids-peak/ |
| PyAV (`av`) | 14.0.0 | BSD-3-Clause wrapper over LGPL-2.1 FFmpeg | https://github.com/PyAV-Org/PyAV |

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

**Authoritative per-build list**: The imagecodecs wheel includes a `LICENSES/` subdirectory inside `site-packages/imagecodecs/` on the build machine. Contents of that directory should be copied verbatim into `docs/licenses/codecs/` in the installer before ship, to capture any additional codecs enabled in the specific wheel version.

## Bundled binaries (not pip-installed)

These ship inside the application folder after PyInstaller builds the Windows installer:

| Binary | License | Upstream | How it's bundled |
|---|---|---|---|
| FFmpeg shared libraries (`avcodec`, `avformat`, `avutil`, `swscale`, `swresample`) | LGPL-2.1-or-later | https://ffmpeg.org/download.html | Dynamically linked from PyAV. LumaViewPro ships PyAV's stock PyPI wheels, whose FFmpeg binaries are built LGPL-only (no `--enable-gpl` components) per PyAV's upstream wheel build configuration. DLLs live in the application's `av.libs/` folder (or equivalent) and are user-replaceable per LGPL §6. |
| SDL2 + SDL2_image / SDL2_mixer / SDL2_ttf | zlib license | https://www.libsdl.org/ | Bundled via `kivy_deps.sdl2` and included in the PyInstaller `.spec` via `Tree(sdl2.dep_bins)`. See `docs/licenses/LICENSE.*.txt` for each sub-component. |
| GLEW | Modified BSD / MIT | http://glew.sourceforge.net/ | Bundled via `kivy_deps.glew`. |
| OpenBLAS | BSD-3-Clause | https://www.openblas.net/ | Bundled inside NumPy/SciPy wheels. |

## Separately installed by end user (not bundled)

These are proprietary SDKs required for specific camera hardware. Each has its own EULA, accepted by the user during its own installer. LumaViewPro's Python wrappers (`pypylon`, `ids-peak`) are BSD-3 and are bundled, but the proprietary SDKs themselves are not.

| SDK | License | Upstream |
|---|---|---|
| Basler Pylon SDK | Proprietary (Basler EULA) | https://docs.baslerweb.com/pylon-software-suite |
| IDS Peak SDK | Proprietary (IDS EULA) | https://en.ids-imaging.com/ids-peak.html |

The Windows Bundle installer *chains* Basler's signed Pylon MSI so the Basler EULA is presented during install. IDS Peak is always installed separately by the end user.

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
- `docs/licenses/LICENSE.FLAC.txt`, `LICENSE.freetype.txt`, `LICENSE.harfbuzz.txt`, `LICENSE.jpeg.txt`, `LICENSE.modplug.txt`, `LICENSE.mpg123.txt`, `LICENSE.ogg-vorbis.txt`, `LICENSE.opus.txt`, `LICENSE.opusfile.txt`, `LICENSE.png.txt`, `LICENSE.tiff.txt`, `LICENSE.webp.txt`, `LICENSE.zlib.txt` — SDL2 sub-component licenses, pre-existing.
