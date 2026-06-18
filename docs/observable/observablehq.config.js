import katex from "katex";
import texmath from "markdown-it-texmath";

export default {
  title: "Strom — Smart Heating Optimisation",
  base: process.env.OBSERVABLE_BASE || "/",
  root: "src",
  pages: [],
  theme: ["dark", "slate"],
  toc: true,
  pager: false,
  style: "style.css",
  head: '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css">',
  markdownIt: (md) => md.use(texmath, { engine: katex, delimiters: "dollars" }),
};
