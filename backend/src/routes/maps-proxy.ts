import { Router } from "express";
import { config } from "../config";

export const mapsProxyRouter = Router();

const GMAPS_KEY = process.env.GOOGLE_MAPS_KEY ?? "";

/** Proxy: GET /v1/maps/geocode?address=... */
mapsProxyRouter.get("/geocode", async (req, res) => {
  if (!GMAPS_KEY) return res.status(500).json({ error: "GOOGLE_MAPS_KEY not set on server" });

  const address = String(req.query.address ?? "");
  if (!address) return res.status(400).json({ error: "address required" });

  try {
    const url = `https://maps.googleapis.com/maps/api/geocode/json?` +
      new URLSearchParams({
        address,
        key: GMAPS_KEY,
        bounds: "40.3,-80.2|40.6,-79.7"
      });

    const gRes = await fetch(url);
    const data = await gRes.json();
    return res.json(data);
  } catch (e) {
    return res.status(502).json({ error: "Geocode proxy failed" });
  }
});

/** Proxy: GET /v1/maps/autocomplete?input=... — Places Autocomplete for multiple suggestions */
mapsProxyRouter.get("/autocomplete", async (req, res) => {
  if (!GMAPS_KEY) return res.status(500).json({ error: "GOOGLE_MAPS_KEY not set on server" });

  const input = String(req.query.input ?? "");
  if (!input) return res.status(400).json({ error: "input required" });

  try {
    const url = `https://maps.googleapis.com/maps/api/place/autocomplete/json?` +
      new URLSearchParams({
        input,
        key: GMAPS_KEY,
        // Bias results toward Pittsburgh
        location: "40.4406,-79.9959",
        radius: "30000",
        components: "country:us"
      });

    const gRes = await fetch(url);
    const data = await gRes.json();
    return res.json(data);
  } catch (e) {
    return res.status(502).json({ error: "Autocomplete proxy failed" });
  }
});

/** Proxy: GET /v1/maps/place-details?place_id=... — resolve a place_id to lat/lng */
mapsProxyRouter.get("/place-details", async (req, res) => {
  if (!GMAPS_KEY) return res.status(500).json({ error: "GOOGLE_MAPS_KEY not set on server" });

  const placeId = String(req.query.place_id ?? "");
  if (!placeId) return res.status(400).json({ error: "place_id required" });

  try {
    const url = `https://maps.googleapis.com/maps/api/geocode/json?` +
      new URLSearchParams({ place_id: placeId, key: GMAPS_KEY });

    const gRes = await fetch(url);
    const data = await gRes.json();
    return res.json(data);
  } catch (e) {
    return res.status(502).json({ error: "Place details proxy failed" });
  }
});

/** Proxy: GET /v1/maps/directions?origin=lat,lng&destination=lat,lng */
mapsProxyRouter.get("/directions", async (req, res) => {
  if (!GMAPS_KEY) return res.status(500).json({ error: "GOOGLE_MAPS_KEY not set on server" });

  const origin = String(req.query.origin ?? "");
  const destination = String(req.query.destination ?? "");
  if (!origin || !destination) return res.status(400).json({ error: "origin and destination required" });

  try {
    const url = `https://maps.googleapis.com/maps/api/directions/json?` +
      new URLSearchParams({
        origin,
        destination,
        key: GMAPS_KEY,
        mode: "driving",
        departure_time: "now"
      });

    const gRes = await fetch(url);
    const data = await gRes.json();
    return res.json(data);
  } catch (e) {
    return res.status(502).json({ error: "Directions proxy failed" });
  }
});
