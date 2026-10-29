// SciQuest offline MathJax fallback.
// Full MathJax can be swapped in at this path for production offline rendering.
// This lightweight fallback prevents dashboards from depending on a CDN and leaves
// TeX equations visible as readable text when the full renderer is unavailable.
window.MathJax = window.MathJax || { typesetPromise: function(){ return Promise.resolve(); } };
document.documentElement.dataset.mathjax = 'sciquest-offline-fallback';
