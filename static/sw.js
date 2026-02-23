// Flow service worker — minimal, just enables PWA installability
const CACHE = 'flow-v6';
const ASSETS = ['/', '/static/manifest.json', '/static/icon-192.png', '/static/icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE).map(k => caches.delete(k))
    ))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  // Network-first for everything (we want fresh content)
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
