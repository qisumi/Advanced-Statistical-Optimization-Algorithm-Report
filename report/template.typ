// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let project(title: "",subtitle:"", authors: (), logo: none, body) = {

  // Set the document's basic properties.
  set document(author: authors.map(a => a.name), title: title)
  set page(numbering: "1 / 1", number-align: center)
  
  // 正文宋体五号字
  set text(font: "Source Sans Pro", lang: "zh",size: 10.5pt)
  
  set heading(numbering: "1.1")

  // 一级标题剧中
  show heading.where(
    level: 1
  ): it => [
    #v(0.5em)
    #set align(center)
    #it
    #v(0.5em)
  ]
  

  // 封面页.
  // LOGO "logo.png"`.
  v(0.6fr)
  if logo != none {
    align(right, image(logo, width: 60%))
  }
  v(0fr)
  // 主标题


  text(3.5em, title)
  v(1fr)
  // 副标题
  text(1.5em, subtitle)
  v(9.6fr)

  // Author information.
  pad(
    top: 0.7em,
    grid(
      columns: (1fr,) * calc.min(3, authors.len()),
      gutter: 1em,
      ..authors.map(author => align(right)[
        #text(14pt,[*#author.name* \
        #author.email])
        
      ]),
    ),
  )

  v(2.4fr)
  pagebreak()
  


  // 目录
  outline(depth: 3, indent: true)
  outline(
    title: [插图目录],
    target: figure.where(kind: image),
  )
  outline(
    title: [表格目录],
    target: figure.where(kind: table),
  )
  pagebreak()
  

  // 链接样式
  show ref: underline 
  show ref: set text(blue)
  show link: underline
  show link: set text(blue)
  

  // Main body.
  set par(
    first-line-indent: 2em, 
    justify: true,
    leading: 12pt,
  )

  set math.equation(numbering: "(1)")

  body
}