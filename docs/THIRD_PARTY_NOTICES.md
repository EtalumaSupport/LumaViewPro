# LumaViewPro — Third-Party Notices

LumaViewPro is distributed under the MIT License (see `docs/LICENSE`). It bundles and uses software from the third parties listed below. Each entry identifies the component, the license under which it is distributed, and the upstream source. Full license texts for each license family are provided in `docs/licenses/`.

This NOTICE file is maintained by Etaluma, Inc. in accordance with Apache-2.0 §4(d) and comparable attribution requirements from BSD, LGPL, and GPL-with-Classpath-Exception licenses that apply to bundled components.

**Last updated**: 2026-04-17 (for LumaViewPro 4.0.0-beta)
**How to regenerate**: On the Windows build machine, after `pip install -r requirements.txt`, run `pip-licenses --with-license-file --with-notice-file --format=markdown` to produce an authoritative per-wheel listing and reconcile with this file.

---

## Bundled Python packages (pip-installed and shipped inside the application folder)

| Package | Version | License | Upstream |
|---|---|---|---|
| Kivy | 2.3.1 | MIT | https://kivy.org/ |
| kivy-deps (sdl2, glew) | bundled with Kivy | zlib / MIT (SDL2 family) | https://www.libsdl.org/ |
| NumPy | 2.4.3 | BSD-3-Clause | https://numpy.org/ |
| SciPy | 1.17.1 | BSD-3-Clause | https://scipy.org/ |
| Matplotlib | 3.10.8 | Matplotlib License (PSF-based, BSD-compatible) | https://matplotlib.org/ |
| pandas | 3.0.1 | BSD-3-Clause | https://pandas.pydata.org/ |
| scikit-image | 0.26.0 | BSD-3-Clause | https://scikit-image.org/ |
| opencv-python-headless | 4.13.0.92 | Apache-2.0 | https://opencv.org/ |
| tifffile | 2026.3.3 | BSD-3-Clause | https://github.com/cgohlke/tifffile |
| imagecodecs | 2026.3.6 | BSD-3-Clause (wrapper) | https://github.com/cgohlke/imagecodecs |
| xarray | 2026.2.0 | Apache-2.0 | https://xarray.dev/ |
| pyserial | 3.5 | BSD-3-Clause | https://github.com/pyserial/pyserial |
| psutil | 7.2.2 | BSD-3-Clause | https://github.com/giampaolo/psutil |
| Numba | 0.64.0 | BSD-2-Clause | https://numba.pydata.org/ |
| llvmlite (numba dep) | bundled with numba | BSD-2-Clause (wrapper); LLVM is Apache-2.0 with LLVM Exceptions | https://github.com/numba/llvmlite |
| userpaths | 0.1.3 | MIT | https://pypi.org/project/userpaths/ |
| pypylon | 4.2.0 | BSD-3-Clause (Python wrapper only) | https://github.com/basler/pypylon |
| ids-peak | 1.13.0.0.6 | BSD-3-Clause (Python wrapper only) | https://pypi.org/project/ids-peak/ |
| PyAV (`av`) | ≥14.0.0 | BSD-3-Clause wrapper over LGPL-2.1 FFmpeg | https://github.com/PyAV-Org/PyAV |
| JPype1 | 1.6.0 | Apache-2.0 | https://github.com/jpype-project/jpype |
| pyimagej | 1.7.0 | Apache-2.0 | https://github.com/imagej/pyimagej |
| scyjava | 1.12.2 | BSD-2-Clause | https://github.com/scijava/scyjava |
| jgo | 1.0.6 | BSD-2-Clause | https://github.com/scijava/jgo |

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
| FFmpeg shared libraries (`avcodec`, `avformat`, `avutil`, `swscale`, `swresample`) | LGPL-2.1-or-later | https://ffmpeg.org/download.html | Dynamically linked from PyAV; DLLs live in the application's `av.libs/` folder (or equivalent) and are user-replaceable per LGPL §6. |
| SDL2 + SDL2_image / SDL2_mixer / SDL2_ttf | zlib license | https://www.libsdl.org/ | Bundled via `kivy_deps.sdl2` and included in the PyInstaller `.spec` via `Tree(sdl2.dep_bins)`. See `docs/licenses/LICENSE.*.txt` for each sub-component. |
| GLEW | Modified BSD / MIT | http://glew.sourceforge.net/ | Bundled via `kivy_deps.glew`. |
| OpenBLAS | BSD-3-Clause | https://www.openblas.net/ | Bundled inside NumPy/SciPy wheels. |
| LLVM | Apache-2.0 with LLVM Exceptions | https://llvm.org/ | Bundled inside llvmlite (numba dep). |
| Apache Maven 3.9.8 (Windows Bundle installer only) | Apache-2.0 | https://maven.apache.org/ | Copied under the install directory for PyImageJ support. The Maven `NOTICE` file from the upstream archive MUST be included verbatim — see the **Maven NOTICE** section below. |
| Amazon Corretto 8 JDK (Windows Bundle installer only) | GPL-2.0 with the Classpath Exception | https://github.com/corretto/corretto-8 | Chained into the `setup.exe` bundle via Amazon's signed, unmodified MSI. Installed to its own directory. |

## Separately installed by end user (not bundled)

These are proprietary SDKs required for specific camera hardware. Each has its own EULA, accepted by the user during its own installer. LumaViewPro's Python wrappers (`pypylon`, `ids-peak`) are BSD-3 and are bundled, but the proprietary SDKs themselves are not.

| SDK | License | Upstream |
|---|---|---|
| Basler Pylon SDK | Proprietary (Basler EULA) | https://docs.baslerweb.com/pylon-software-suite |
| IDS Peak SDK | Proprietary (IDS EULA) | https://en.ids-imaging.com/ids-peak.html |

The Windows Bundle installer *chains* Basler's signed Pylon MSI so the Basler EULA is presented during install. IDS Peak is always installed separately by the end user.

---

## Maven NOTICE (Apache-2.0 §4(d) carry-forward)

The following NOTICE content is carried forward verbatim from Apache Maven 3.9.8's upstream `NOTICE` file (downloaded from `https://archive.apache.org/dist/maven/maven-3/3.9.8/binaries/apache-maven-3.9.8-bin.tar.gz` on 2026-04-17):

```
Apache Maven Distribution
Copyright 2001-2024 The Apache Software Foundation


This product includes software developed at
The Apache Software Foundation (http://www.apache.org/).
This software bundles the following NOTICE files from third party library providers:

META-INF/NOTICE in archive lib/guice-5.1.0.jar
Google Guice - Core Library
Copyright 2006-2022 Google, Inc.
This product includes software developed at
The Apache Software Foundation (http://www.apache.org/).

META-INF/NOTICE in archive lib/plexus-utils-3.2.1.jar
This product includes software developed by the Indiana University
 Extreme! Lab (http://www.extreme.indiana.edu/).
This product includes software developed by
The Apache Software Foundation (http://www.apache.org/).
This product includes software developed by
ThoughtWorks (http://www.thoughtworks.com).
This product includes software developed by
javolution (http://javolution.org/).
This product includes software developed by
Rome (https://rome.dev.java.net/).

about.html in archive lib/org.eclipse.sisu.inject-0.3.5.jar

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1"/>
<title>About org.eclipse.sisu.inject</title>
</head>
<body lang="EN-US">
<h2>About org.eclipse.sisu.inject</h2>
 
<p>November 5, 2013</p>	
<h3>License</h3>

<p>The Eclipse Foundation makes available all content in this plug-in ("Content").  Unless otherwise 
indicated below, the Content is provided to you under the terms and conditions of the
Eclipse Public License Version 1.0 ("EPL").  A copy of the EPL is available 
at <a href="http://www.eclipse.org/legal/epl-v10.html">http://www.eclipse.org/legal/epl-v10.html</a>.
For purposes of the EPL, "Program" will mean the Content.</p>

<p>If you did not receive this Content directly from the Eclipse Foundation, the Content is 
being redistributed by another party ("Redistributor") and different terms and conditions may
apply to your use of any object code in the Content.  Check the Redistributor's license that was 
provided with the Content.  If no such license exists, contact the Redistributor.  Unless otherwise
indicated below, the terms and conditions of the EPL still apply to any source code in the Content
and such source code may be obtained at <a href="http://www.eclipse.org/">http://www.eclipse.org</a>.</p>

<h3>Third Party Content</h3>
<p>The Content includes items that have been sourced from third parties as set 
out below. If you did not receive this Content directly from the Eclipse Foundation, 
the following is provided for informational purposes only, and you should look 
to the Redistributor's license for terms and conditions of use.</p>

<h4>ASM 4.1</h4>
<p>The plug-in includes software developed by the ObjectWeb consortium as part
of the ASM project at <a href="http://asm.ow2.org/">http://asm.ow2.org/</a>.</p>

<p>A subset of ASM is re-packaged within the source and binary of the plug-in (org.eclipse.sisu.space.asm.*)
to avoid version collisions with other usage and is also available from the plug-in's github repository.</p>

<p>Your use of the ASM code is subject to the terms and conditions of the ASM License
below which is also available at <a href="http://asm.ow2.org/license.html">http://asm.ow2.org/license.html</a>.</p>

<blockquote><pre>
Copyright (c) 2000-2011 INRIA, France Telecom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
</pre></blockquote>

</body>
</html>
```

## Apache OpenJDK (Corretto) NOTICE

Amazon Corretto is distributed as Amazon-signed MSI installers sourced from https://github.com/corretto/corretto-8. Corretto's license (`docs/licenses/LICENSE.GPL-2.0-Classpath-Exception.txt`) and Amazon's attribution materials are carried inside Amazon's MSI. Etaluma, Inc. does not repackage or modify Corretto.

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
- `docs/licenses/LICENSE.GPL-2.0-Classpath-Exception.txt`
- `docs/licenses/LICENSE.FLAC.txt`, `LICENSE.freetype.txt`, `LICENSE.harfbuzz.txt`, `LICENSE.jpeg.txt`, `LICENSE.modplug.txt`, `LICENSE.mpg123.txt`, `LICENSE.ogg-vorbis.txt`, `LICENSE.opus.txt`, `LICENSE.opusfile.txt`, `LICENSE.png.txt`, `LICENSE.tiff.txt`, `LICENSE.webp.txt`, `LICENSE.zlib.txt` — SDL2 sub-component licenses, pre-existing.
