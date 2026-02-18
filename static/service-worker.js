const CACHE_NAME = 'crop-app-v2'; // ✅ Version bumped
const ASSETS_TO_CACHE = [
  '/',
  '/diagnose',
  '/static/manifest.json',
  '/static/icons8-chaff-48.png',
  '/static/tfjs_model/model.json',
  'https://cdn.tailwindcss.com',
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js' 
];

self.addEventListener('install', (event) => {
  self.skipWaiting(); // Force activation immediately
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('✅ Caching Shell Assets...');
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

// 🔥 CRITICAL FIX: Delete old 'v1' caches so we don't fill up storage
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cache) => {
                    if (cache !== CACHE_NAME) {
                        console.log('🧹 Clearing Old Cache:', cache);
                        return caches.delete(cache);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', (event) => {
  const url = event.request.url;

  // STRATEGY 1: Cache First (For Model & Shards)
  if (url.includes('group1-shard') || url.includes('tfjs_model') || url.includes('tf.min.js')) {
    event.respondWith(
      caches.match(event.request).then((cachedResponse) => {
        if (cachedResponse) {
          return cachedResponse;
        }
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
  else if (event.request.mode === 'navigate') {
     event.respondWith(
        fetch(event.request).catch(() => caches.match(event.request))
     );
  }
  // STRATEGY 3: Stale-While-Revalidate (For everything else)
  else {
    event.respondWith(caches.match(event.request).then((response) => response || fetch(event.request)));
  }
});