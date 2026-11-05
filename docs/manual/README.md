# The LumaViewPro user manual

The source of the Lumascope Operator's Manual: a Quarto book that renders
to PDF through Quarto's bundled Typst.

## Install Quarto 1.11.9

The project refuses any other version (`quarto-required` in `_quarto.yml`).
Download from the [v1.11.9 release](https://github.com/quarto-dev/quarto-cli/releases/tag/v1.11.9)
and check the download against `quarto-1.11.9-checksums.txt` from the same release.

- macOS: `quarto-1.11.9-macos.tar.gz`; `shasum -a 256` it, unpack it, and put its `bin/quarto` on your PATH (or run `quarto-1.11.9-macos.pkg`).
- Windows: `quarto-1.11.9-win.msi`; check it with `certutil -hashfile quarto-1.11.9-win.msi SHA256`, then run it.
- Linux: `quarto-1.11.9-linux-amd64.tar.gz`; `sha256sum` it, unpack it, and put its `bin/quarto` on your PATH.

## Build the PDF

```
quarto render docs/manual --to typst
```

The PDF is written to `docs/manual/_book/LumaViewPro-Manual.pdf`, which git
ignores. It is distributed as a separate download, not committed.

## Layout

- `*.qmd`: the chapters, in the order `_quarto.yml` lists them.
- `images/`: every figure, at print size.
- `style/`: the look (`etaluma.typ`), the partial that applies it, and `guilabel.lua`, which prints a marked control or channel in bold and a setting in code type.
- `fonts/`: TeX Gyre Heros, under the GUST Font License (`fonts/GUST-FONT-LICENSE.txt`).

Mark a control `[Image Mode]{.guilabel}`, a channel drawer `[488 nm Excitation]{.channel}`
and a setting `[protocol.autogain.target_brightness]{.setting}`.
`tests/guards/test_the_manual_names_what_the_code_has.py` checks each against the code.
