const CACHE_NAME = 'crop-app-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/diagnose',
  '/static/manifest.json',
  '/static/icons8-chaff-48.png',
  '/static/tfjs_model/model.json',
  'https://cdn.tailwindcss.com',
  // LOCKED VERSION for stability (don't use @latest in production!)
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js' 
];

self.addEventListener('install', (event) => {
  self.skipWaiting(); // Force activation immediately
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS_TO_CACHE)));
});

self.addEventListener('activate', (event) => {
    event.waitUntil(self.clients.claim()); // Take control of all pages immediately
});

self.addEventListener('fetch', (event) => {
  const url = event.request.url;

  // STRATEGY 1: Cache First (For Model & Shards)
  // If we have the model, use it. Don't waste data checking for updates.
  if (url.includes('group1-shard') || url.includes('tfjs_model') || url.includes('tf.min.js')) {
    event.respondWith(
      caches.match(event.request).then((cachedResponse) => {
        if (cachedResponse) {
          return cachedResponse; // Return cache immediately, NO network request
        }
        // If not in cache, fetch it and cache it
        return fetch(event.request).then((networkResponse) => {
          return caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, networkResponse.clone());
            return networkResponse;
          });
        });
      })
    );
  } 
  // STRATEGY 2: Network First (For HTML Pages)
  // Ensures user always sees latest UI if online, falls back to cache if offline
  else if (event.request.mode === 'navigate') {
     event.respondWith(
        fetch(event.request).catch(() => caches.match(event.request))
     );
  }
  // STRATEGY 3: Stale-While-Revalidate (For everything else like CSS/Icons)
  else {
    event.respondWith(caches.match(event.request).then((response) => response || fetch(event.request)));
  }
});