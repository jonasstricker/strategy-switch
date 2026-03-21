// Service Worker for Strategy-Switch PWA (GitHub Pages)
const CACHE_NAME = 'strat-switch-v2';
const CACHED_URLS = [
  '.',
  'style.css',
  'manifest.json',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(CACHED_URLS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  // Always fetch data fresh (no caching for JSON)
  if (event.request.url.includes('/data/')) {
    event.respondWith(fetch(event.request));
    return;
  }
  // Network first, fall back to cache for static assets
  event.respondWith(
    fetch(event.request)
      .then(resp => {
        const clone = resp.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        return resp;
      })
      .catch(() => caches.match(event.request))
  );
});
