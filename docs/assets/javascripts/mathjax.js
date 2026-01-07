window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams',
    packages: {'[+]': ['boldsymbol', 'ams', 'newcommand', 'configmacros']},
    macros: {
      // Define common macros if needed
      bm: ["\\boldsymbol{#1}", 1]
    }
  },
  loader: {
    load: ['[tex]/boldsymbol', '[tex]/ams', '[tex]/newcommand', '[tex]/configmacros']
  },
  options: {
    ignoreHtmlClass: ".*",
    processHtmlClass: "arithmatex|md-nav|md-sidebar|md-content"
  },
  startup: {
    ready: () => {
      MathJax.startup.defaultReady();
      // Re-typeset when details elements are opened
      document.querySelectorAll('details').forEach(details => {
        details.addEventListener('toggle', () => {
          if (details.open) {
            MathJax.typesetPromise([details]);
          }
        });
      });
    }
  }
};

// For MkDocs Material instant loading
if (typeof document$ !== 'undefined') {
  document$.subscribe(() => {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise().then(() => {
      // Also process navigation elements
      const navElements = document.querySelectorAll('.md-nav, .md-sidebar, .md-header');
      if (navElements.length > 0) {
        MathJax.typesetPromise(Array.from(navElements));
      }
    });
  });
}
