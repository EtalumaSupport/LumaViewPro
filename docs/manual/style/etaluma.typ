// The Etaluma manual look: a sans face throughout, black numbered
// headings, a photo cover with the teal rule, a grey running footer.

#let teal = rgb("#9ACCCC")
#let grey = rgb("#8C8C8C")

#let cover(title, subtitle, version) = page(margin: (x: 1in, top: 0.6in, bottom: 0.8in), footer: none)[
  #align(center, image("../images/cover.png", width: 100%))
  #v(1.2em)
  #text(size: 26pt, tracking: 0.04em)[#title]
  #if subtitle != none [#v(-0.4em) #text(size: 13pt)[#subtitle]]
  #v(-0.6em)
  #block(width: 70%, height: 7pt, fill: gradient.linear(teal, white))
  #v(0.8em)
  #text(size: 13pt, tracking: 0.05em)[Version #version]
  #v(1fr)
  #grid(columns: (1fr, auto), align: (left + bottom, right + bottom),
    text(size: 9.5pt)[
      For assistance, please email \
      #link("mailto:support@etaluma.com")[#text(fill: blue)[support\@etaluma.com]] \
      #v(0.4em)
      Etaluma, Inc. \
      Carlsbad, California \
      #link("https://www.etaluma.com")[#text(fill: blue)[www.etaluma.com]]
    ],
    image("../images/etaluma-logo.png", width: 2.4in),
  )
]

#let manual(title: none, subtitle: none, version: "", outline-depth: 2, body) = {
  set document(title: title)
  set text(font: "TeX Gyre Heros", size: 10.5pt, lang: "en")
  set par(justify: false, leading: 0.62em, spacing: 1.1em)
  show link: set text(fill: black)
  show raw: set text(font: "DejaVu Sans Mono", size: 0.92em)

  set heading(numbering: "1.1")
  show heading: set text(weight: "bold")
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    block(above: 0pt, below: 1.2em, text(size: 17pt)[
      #if it.numbering != none { counter(heading).display(); h(0.9em) }#it.body
    ])
  }
  show heading.where(level: 2): it => block(above: 1.6em, below: 0.8em, text(size: 12pt)[
    #if it.numbering != none { counter(heading).display(); h(0.9em) }#it.body
  ])
  show heading.where(level: 3): set text(size: 10.5pt)

  // Figures and tables numbered within their chapter (Figure 2.1); captions small and centred.
  set figure(numbering: n => numbering("1.1", counter(heading).get().first(), n))
  show figure.caption: it => text(size: 9pt)[#it.supplement #context it.counter.display(it.numbering): #it.body]
  set figure(gap: 1em)
  show figure: set block(above: 1.5em, below: 1.5em)
  set table(stroke: none, inset: (x: 8pt, y: 5pt))

  set page(paper: "us-letter", margin: (x: 1in, top: 1in, bottom: 1.1in),
    footer: context text(size: 9pt, fill: grey)[
      LumaViewPro #version Manual #h(1fr) #text(fill: black)[#counter(page).display()]
    ])

  cover(title, subtitle, version)

  // Contents: chapters bold with their page, sections indented on a dotted leader.
  show outline.entry.where(level: 1): it => {
    v(0.9em, weak: true)
    strong(link(it.element.location(), it.indented(it.prefix(), [#it.body() #h(1fr) #it.page()])))
  }
  outline(title: text(size: 17pt, weight: "bold")[Contents], depth: outline-depth, indent: 1.4em)

  body
}

// orange-book's filter emits these for parts and appendices.
#let part(title) = heading(level: 1, numbering: none, title)
#let chapter(title) = heading(level: 1, title)
#let appendices(title, hide-parent: false, body) = {
  counter(heading).update(0)
  set heading(numbering: "A.1")
  body
}

// The callout, restyled: a teal or amber bar on the left, the title bold.
#let callout(body: [], title: "Callout", background_color: none, icon: none, icon_color: black, body_background_color: white) = block(
  width: 100%, breakable: false, inset: (left: 10pt, y: 6pt), above: 1.2em, below: 1.2em,
  stroke: (left: 3pt + icon_color),
)[
  #if title != none and title != [] { strong(title); linebreak() }
  #body
]
