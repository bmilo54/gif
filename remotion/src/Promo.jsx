import React from "react";
import { Lottie } from "@remotion/lottie";
import {
  AbsoluteFill,
  Img,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

import sparkleData from "./lottie/sparkle.json";
import { computeEffectStyle, needsLottie } from "./effects/registry";

// ---------------------------------------------------------------------------
// Asset helpers
// ---------------------------------------------------------------------------

function assetSrc(src) {
  if (!src) return null;
  if (
    src.startsWith("http://") ||
    src.startsWith("https://") ||
    src.startsWith("file:") ||
    src.startsWith("data:")
  ) {
    return src;
  }
  return staticFile(src);
}

// ---------------------------------------------------------------------------
// Lottie preset map
// ---------------------------------------------------------------------------

const LOTTIE_PRESETS = {
  sparkle: sparkleData,
};

// ---------------------------------------------------------------------------
// Pixel-box helpers
// ---------------------------------------------------------------------------

function boxPixels(region, canvasW, canvasH) {
  const left   = region.x * canvasW;
  const top    = region.y * canvasH;
  const width  = region.width * canvasW;
  const height = region.height * canvasH;
  const src    = (region.source || "").toLowerCase();
  const radius =
    src === "button"
      ? Math.max(4, height / 2)
      : Math.min(16, Math.min(width, height) * 0.12);
  return { left, top, width, height, radius };
}

// ---------------------------------------------------------------------------
// CharacterLayer
// Renders one SAM-segmented character PNG (transparent bg) positioned by bbox.
// ---------------------------------------------------------------------------

function CharacterLayer({ character, canvasW, canvasH, frame, dur }) {
  const { src, bbox, effects } = character;
  const imgSrc = assetSrc(src);
  if (!imgSrc || !bbox || !(bbox.width > 0) || !(bbox.height > 0)) return null;

  const { left, top, width, height } = boxPixels(bbox, canvasW, canvasH);
  const effectStyle = computeEffectStyle(effects || [], frame, dur);

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        overflow: "hidden",
        pointerEvents: "none",
        transformOrigin: "center center",
        ...effectStyle,
      }}
    >
      <Img
        src={imgSrc}
        style={{ width: "100%", height: "100%", objectFit: "fill" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// LottieOverlay
// Renders a Lottie animation covering a region (mix-blend: screen).
// ---------------------------------------------------------------------------

function LottieOverlay({ region, animationData, canvasW, canvasH, dur }) {
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);
  const ip = Number(animationData.ip) || 0;
  const op = Number(animationData.op) || 60;
  const playbackRate = (op - ip) / Math.max(dur, 1);

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        overflow: "hidden",
        borderRadius: radius,
        pointerEvents: "none",
        mixBlendMode: "screen",
      }}
    >
      <Lottie
        animationData={animationData}
        playbackRate={playbackRate}
        loop
        style={{ width: "100%", height: "100%" }}
        preserveAspectRatio="none"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// RegionEffectLayer
// For non-person regions (card, button, ocr …): renders the poster slice with
// its own CSS effects, then Lottie overlays if requested.
// ---------------------------------------------------------------------------

function RegionEffectLayer({ region, posterSrc, canvasW, canvasH, frame, dur }) {
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);
  const effects  = region.effects || [];
  const effectStyle = computeEffectStyle(effects, frame, dur);

  return (
    <>
      {/* Poster slice with motion / lighting effects */}
      <div
        style={{
          position: "absolute",
          left,
          top,
          width,
          height,
          overflow: "hidden",
          borderRadius: radius,
          pointerEvents: "none",
          transformOrigin: "center center",
          ...effectStyle,
        }}
      >
        <Img
          src={posterSrc}
          style={{
            position: "absolute",
            left: -left,
            top: -top,
            width: canvasW,
            height: canvasH,
            objectFit: "fill",
          }}
        />
      </div>

      {/* Lottie overlays for sparkle / glow particle effects */}
      {needsLottie(effects) &&
        Object.entries(LOTTIE_PRESETS)
          .filter(([key]) => effects.includes(key))
          .map(([key, data]) => (
            <LottieOverlay
              key={`lottie-${key}-${region.key || ""}`}
              region={region}
              animationData={data}
              canvasW={canvasW}
              canvasH={canvasH}
              dur={dur}
            />
          ))}
    </>
  );
}

// ---------------------------------------------------------------------------
// Promo composition
// ---------------------------------------------------------------------------

const PERSON_SOURCES = new Set(["yolo", "sam"]);

function isPersonRegion(region) {
  const src   = (region.source || "").toLowerCase();
  const label = (region.label  || "").toLowerCase();
  return PERSON_SOURCES.has(src) || label.includes("person");
}

export const Promo = ({ poster, regions, characters }) => {
  const { durationInFrames: dur, width, height } = useVideoConfig();
  const frame = useCurrentFrame();

  const posterSrc  = assetSrc(poster);
  const allRegions = Array.isArray(regions)    ? regions    : [];
  const allChars   = Array.isArray(characters) ? characters : [];

  // UI regions: everything that is NOT a raw yolo/sam person box.
  // Person boxes are already baked into CharacterLayer via the characters prop.
  const uiRegions = allRegions.filter((r) => !isPersonRegion(r));

  return (
    <AbsoluteFill style={{ background: "#000", overflow: "hidden" }}>
      {/* 1. Full poster background */}
      {posterSrc && (
        <Img src={posterSrc} style={{ width, height, objectFit: "fill" }} />
      )}

      {/* 2. SAM-segmented characters — each with its own per-region effects */}
      {allChars.map((char) => (
        <CharacterLayer
          key={`char-${char.index}`}
          character={char}
          canvasW={width}
          canvasH={height}
          frame={frame}
          dur={dur}
        />
      ))}

      {/* 3. UI regions (card, button, ocr, title, prop …) — each with its own effects */}
      {posterSrc &&
        uiRegions.map((region, idx) => (
          <RegionEffectLayer
            key={`region-${region.key || idx}`}
            region={region}
            posterSrc={posterSrc}
            canvasW={width}
            canvasH={height}
            frame={frame}
            dur={dur}
          />
        ))}
    </AbsoluteFill>
  );
};
