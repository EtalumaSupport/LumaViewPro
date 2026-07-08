# LumaViewPro Changelog

## 4.0.0 (in development)

- **API stability**: Lumascope SDK + REST API are PRE-RELEASE. Subject to
  breaking changes in 4.1 / 4.1.5 / 4.2. See `docs/LumascopeSkills.md`
  preface for the migration plan. Internal LumaViewPro use is not
  affected by the freeze trigger.
- **Saved-image bit depth (file-format change)**: full-pixel-depth TIFFs now
  store raw, right-aligned sensor values (a 12-bit frame is `0..4095`) and
  record the true depth in the OME-TIFF `SignificantBits` tag, instead of
  left-justifying the data into the 16-bit container (`value * 16`). This is
  the standard OME-TIFF representation and fixes dim/grainy rendering of 10-bit
  captures and a 16-bit overflow on summed full-depth frames. Read-back
  (Post-Processing video, hyperstacks) scales by the file's `SignificantBits`,
  so existing left-justified files (tagged 16-bit) still read correctly -- no
  migration needed.

Detailed release notes for 4.0.0-betaN tags live in
`LVP_4.0.0_CHANGELOG.md` (release-engineering log).
