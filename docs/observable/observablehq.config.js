import katex from "katex";
import texmath from "markdown-it-texmath";

export default {
  title: "Strom: Smart Heating Optimisation",
  base: process.env.OBSERVABLE_BASE || "/",
  root: "src",
  pages: [],
  theme: "air", // light, neutral base; paper/Tufte aesthetic applied in style.css
  toc: true,
  pager: false,
  style: "style.css",

  // KaTeX stylesheet + shared nav (applies theme early, injects nav bar)
  head: '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css"><script src="https://danielviladrich.github.io/nav.js" defer></script>',

  markdownIt: (md) => md.use(texmath, { engine: katex, delimiters: "dollars" }),
};
