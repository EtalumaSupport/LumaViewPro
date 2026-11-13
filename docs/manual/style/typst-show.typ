#import "style/etaluma.typ": manual, part, chapter, appendices, callout
#show: manual.with(
$if(title)$  title: [$title$],$endif$
$if(subtitle)$  subtitle: [$subtitle$],$endif$
$if(manual-version)$  version: "$manual-version$",$endif$
$if(toc-depth)$  outline-depth: $toc-depth$,$endif$
)
