import React from "react";
import { Lottie } from "@remotion/lottie";
import {
  AbsoluteFill,
  Easing,
  Img,
  interpolate,
  interpolateColors,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

import sparkleData from "./lottie/sparkle.json";
import { computeEffectStyle, needsLottie } from "./effects/registry";

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

const LOTTIE_PRESETS = {
  sparkle: sparkleData,
};

const PERSON_SOURCES = new Set(["yolo", "sam"]);
const UI_SOURCES = new Set(["card", "button", "title", "ocr", "prop", "manual"]);
const PIXEL_MOTION = new Set([
  "float", "float-glow", "breathe", "natural-breathe", "zoom", "zoom-in",
  "bounce", "shake", "wave", "spin", "slide-left", "slide-up",
]);

function isPersonRegion(region) {
  const src = (region.source || "").toLowerCase();
  const label = (region.label || "").toLowerCase();
  return PERSON_SOURCES.has(src) || label.includes("person") || label.includes("character");
}

function isUiRegion(region) {
  const src = (region.source || "").toLowerCase();
  return UI_SOURCES.has(src) && !isPersonRegion(region);
}

function hasEffect(effects, name) {
  return Array.isArray(effects) && effects.includes(name);
}

function boxPixels(region, canvasW, canvasH) {
  const left = region.x * canvasW;
  const top = region.y * canvasH;
  const width = region.width * canvasW;
  const height = region.height * canvasH;
  const src = (region.source || "").toLowerCase();
  const radius =
    src === "button"
      ? Math.max(4, height / 2)
      : Math.min(16, Math.min(width, height) * 0.12);
  return { left, top, width, height, radius };
}

function useLoopWave() {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const duration = Math.max(durationInFrames, 1);
  return interpolate(frame, [0, duration / 2, duration], [0, 1, 0], {
    easing: Easing.inOut(Easing.sin),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
}

function ShineBand({ region, canvasW, canvasH }) {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);
  const sweep = interpolate(frame, [0, Math.max(durationInFrames, 1)], [-25, 125], {
    easing: Easing.inOut(Easing.quad),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

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
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `linear-gradient(115deg, transparent 0%, transparent ${sweep}%, rgba(255, 236, 180, 0) ${sweep}%, rgba(255, 236, 180, 0.72) ${sweep + 8}%, rgba(255, 210, 90, 0) ${sweep + 18}%, transparent 100%)`,
        }}
      />
    </div>
  );
}

function GlowWash({ region, canvasW, canvasH, color, opacity }) {
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);
  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        borderRadius: radius,
        background: color,
        opacity,
        mixBlendMode: "screen",
        pointerEvents: "none",
      }}
    />
  );
}

function RimGlow({ region, canvasW, canvasH, strength }) {
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);
  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        borderRadius: radius,
        boxShadow: `inset 0 0 ${8 + 16 * strength}px rgba(255, 214, 110, ${0.2 + 0.4 * strength})`,
        pointerEvents: "none",
      }}
    />
  );
}

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

function wantsPixelMotion(effects) {
  return (effects || []).some((key) => PIXEL_MOTION.has(key));
}

function OverlayFX({ region, canvasW, canvasH, dur, wave }) {
  const effects = region.effects || [];
  const glowColor = interpolateColors(wave, [0, 1], ["rgba(255, 236, 180, 0.0)", "rgba(255, 236, 180, 0.32)"]);
  const goldColor = interpolateColors(wave, [0, 1], ["rgba(255, 214, 110, 0.0)", "rgba(255, 214, 110, 0.38)"]);

  return (
    <>
      {hasEffect(effects, "glow") || hasEffect(effects, "breathe") || hasEffect(effects, "float") ? (
        <GlowWash
          region={region}
          canvasW={canvasW}
          canvasH={canvasH}
          color={region.color || glowColor}
          opacity={1}
        />
      ) : null}
      {hasEffect(effects, "gold_pulse") ? (
        <GlowWash
          region={region}
          canvasW={canvasW}
          canvasH={canvasH}
          color={goldColor}
          opacity={1}
        />
      ) : null}
      {hasEffect(effects, "shine") ? (
        <ShineBand region={region} canvasW={canvasW} canvasH={canvasH} />
      ) : null}
      {hasEffect(effects, "rim") || hasEffect(effects, "breathe") ? (
        <RimGlow region={region} canvasW={canvasW} canvasH={canvasH} strength={wave} color={region.color || glowColor} />
      ) : null}
      {needsLottie(effects)
        ? Object.entries(LOTTIE_PRESETS)
            .filter(([key]) => effects.includes(key))
            .map(([key, data]) => (
              <LottieOverlay
                key={`lottie-${key}`}
                region={region}
                animationData={data}
                canvasW={canvasW}
                canvasH={canvasH}
                dur={dur}
              />
            ))
        : null}
    </>
  );
}

function UiLayer({ region, posterSrc, canvasW, canvasH, frame, dur, wave }) {
  const effects = region.effects || [];
  const motion = wantsPixelMotion(effects);
  const effectStyle = motion ? computeEffectStyle(effects, frame, dur) : {};
  const { left, top, width, height, radius } = boxPixels(region, canvasW, canvasH);

  return (
    <>
      {motion ? (
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
      ) : null}
      <OverlayFX region={region} canvasW={canvasW} canvasH={canvasH} dur={dur} wave={wave} />
    </>
  );
}

function CharacterLayer({ character, canvasW, canvasH, frame, dur, wave }) {
  const effects = character.effects || [];
  const motion = wantsPixelMotion(effects);
  const color = character.color || "#ffecb4";
  const effectStyle = motion ? computeEffectStyle(effects, frame, dur, color) : {};
  
  const left = character.bbox.x * canvasW;
  const top = character.bbox.y * canvasH;
  const width = character.bbox.width * canvasW;
  const height = character.bbox.height * canvasH;

  const hasGlow = hasEffect(effects, "glow") || hasEffect(effects, "breathe") || hasEffect(effects, "natural-breathe");
  const hasRim = hasEffect(effects, "rim");
  
  let filterStyle = "";
  if (hasGlow || hasRim) {
    const spread = 8 + 12 * wave;
    filterStyle = `drop-shadow(0px 0px ${spread}px ${color})`;
  }

  return (
    <div
      style={{
        position: "absolute",
        left,
        top,
        width,
        height,
        pointerEvents: "none",
        transformOrigin: "center center",
        ...effectStyle,
      }}
    >
      <Img
        src={assetSrc(character.src)}
        style={{ 
          width: "100%", 
          height: "100%", 
          objectFit: "fill",
          filter: filterStyle || undefined
        }}
      />
    </div>
  );
}

export const Promo = ({ poster, regions, characters }) => {
  const { durationInFrames: dur, width, height } = useVideoConfig();
  const frame = useCurrentFrame();
  const wave = useLoopWave();
  const posterSrc = assetSrc(poster);
  const allRegions = Array.isArray(regions) ? regions : [];
  const allChars = Array.isArray(characters) ? characters : [];
  
  const ui = allRegions.filter(isUiRegion);
  
  // Also keep people array for fallback if SAM didn't run
  const people = allRegions.filter(isPersonRegion);

  return (
    <AbsoluteFill style={{ background: "#000", overflow: "hidden" }}>
      {posterSrc ? (
        <Img src={posterSrc} style={{ width, height, objectFit: "fill" }} />
      ) : null}

      {allChars.length > 0 ? (
        allChars.map((char) => (
          <CharacterLayer
            key={`char-${char.index}`}
            character={char}
            canvasW={width}
            canvasH={height}
            frame={frame}
            dur={dur}
            wave={wave}
          />
        ))
      ) : (
        people.map((region, idx) => (
          <OverlayFX
            key={`person-${region.key || idx}`}
            region={region}
            canvasW={width}
            canvasH={height}
            dur={dur}
            wave={wave}
          />
        ))
      )}

      {posterSrc
        ? ui.map((region, idx) => (
            <UiLayer
              key={`ui-${region.key || idx}`}
              region={region}
              posterSrc={posterSrc}
              canvasW={width}
              canvasH={height}
              frame={frame}
              dur={dur}
              wave={wave}
            />
          ))
        : null}
    </AbsoluteFill>
  );
};
