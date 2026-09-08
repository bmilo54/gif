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

function isPlaqueCutout(item) {
  const src = (item.source || "").toLowerCase();
  return src === "card" || src === "button" || src === "title";
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
      : src === "card"
        ? Math.min(28, Math.min(width, height) * 0.22)
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
        preserveAspectRatio="xMidYMid meet"
      />
    </div>
  );
}

function wantsPixelMotion(effects) {
  return (effects || []).some((key) => PIXEL_MOTION.has(key));
}

function OverlayFX({ region, canvasW, canvasH, dur, wave, faceWash = true }) {
  const effects = region.effects || [];
  const glowColor = interpolateColors(wave, [0, 1], ["rgba(255, 236, 180, 0.0)", "rgba(255, 236, 180, 0.32)"]);
  const goldColor = interpolateColors(wave, [0, 1], ["rgba(255, 214, 110, 0.0)", "rgba(255, 214, 110, 0.38)"]);
  const wash = faceWash && hasEffect(effects, "glow");
  const goldWash = faceWash && hasEffect(effects, "gold_pulse");
  const rim = faceWash && hasEffect(effects, "rim");

  return (
    <>
      {wash ? (
        <GlowWash
          region={region}
          canvasW={canvasW}
          canvasH={canvasH}
          color={region.color || glowColor}
          opacity={1}
        />
      ) : null}
      {goldWash ? (
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
      {rim ? (
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

function hexToGlow(color, alpha) {
  const raw = String(color || "#ffecb4").replace("#", "");
  if (raw.length !== 6) {
    return `rgba(255, 236, 180, ${alpha})`;
  }
  const r = parseInt(raw.slice(0, 2), 16);
  const g = parseInt(raw.slice(2, 4), 16);
  const b = parseInt(raw.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function CharacterLayer({ character, canvasW, canvasH, frame, dur, wave }) {
  const effects = character.effects || [];
  const motion = wantsPixelMotion(effects);
  const color = character.color || "#ffecb4";
  const effectStyle = motion ? computeEffectStyle(effects, frame, dur, color) : {};
  // Filters on the wrapper recolor skin. Keep motion only; glow stays behind.
  const { filter: _ignoreFilter, ...motionStyle } = effectStyle;

  const left = character.bbox.x * canvasW;
  const top = character.bbox.y * canvasH;
  const width = character.bbox.width * canvasW;
  const height = character.bbox.height * canvasH;

  const wantsHalo = hasEffect(effects, "glow") || hasEffect(effects, "rim");
  const spread = 6 + 8 * wave;
  const halo = hexToGlow(color, 0.35 + 0.2 * wave);
  const src = assetSrc(character.src);
  const fillStyle = { width: "100%", height: "100%", objectFit: "fill" };

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
        ...motionStyle,
      }}
    >
      {wantsHalo && src ? (
        <Img
          src={src}
          style={{
            ...fillStyle,
            position: "absolute",
            inset: 0,
            filter: `drop-shadow(0px 0px ${spread}px ${halo})`,
            opacity: 0.7,
          }}
        />
      ) : null}
      <Img src={src} style={{ ...fillStyle, position: "relative" }} />
    </div>
  );
}

function CutoutLayer({ cutout, canvasW, canvasH, frame, dur, wave }) {
  const effects = cutout.effects || [];
  const motion = wantsPixelMotion(effects);
  const color = cutout.color || "#ffecb4";
  const effectStyle = motion ? computeEffectStyle(effects, frame, dur, color) : {};
  const { filter: _ignoreFilter, ...motionStyle } = effectStyle;
  const left = cutout.bbox.x * canvasW;
  const top = cutout.bbox.y * canvasH;
  const width = cutout.bbox.width * canvasW;
  const height = cutout.bbox.height * canvasH;
  const region = {
    ...cutout,
    x: cutout.bbox.x,
    y: cutout.bbox.y,
    width: cutout.bbox.width,
    height: cutout.bbox.height,
    source: cutout.source || "card",
  };

  const hasGlow =
    hasEffect(effects, "glow") ||
    hasEffect(effects, "gold_pulse") ||
    hasEffect(effects, "rim");
  const spread = 6 + 10 * wave;
  const filterStyle = hasGlow ? `drop-shadow(0px 0px ${spread}px ${color})` : undefined;

  const cutoutSrc = assetSrc(cutout.src);
  const maskStyle = cutoutSrc
    ? {
        WebkitMaskImage: `url(${cutoutSrc})`,
        maskImage: `url(${cutoutSrc})`,
        WebkitMaskSize: "100% 100%",
        maskSize: "100% 100%",
        WebkitMaskRepeat: "no-repeat",
        maskRepeat: "no-repeat",
      }
    : {};

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
        ...motionStyle,
      }}
    >
      <Img
        src={cutoutSrc}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "fill",
          filter: filterStyle,
        }}
      />
      <div style={{ position: "absolute", inset: 0, ...maskStyle }}>
        <OverlayFX
          region={{
            ...region,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
          }}
          canvasW={width}
          canvasH={height}
          dur={dur}
          wave={wave}
          faceWash={false}
        />
      </div>
    </div>
  );
}

function isFrontCutout(item, chars) {
  if (item && item.front) return true;
  const box = item && item.bbox;
  if (!box || !chars || !chars.length) return false;
  const cy = box.y + box.height / 2;
  return chars.some((person) => {
    const p = person.bbox;
    if (!p) return false;
    const overlaps =
      box.x < p.x + p.width &&
      box.x + box.width > p.x &&
      box.y < p.y + p.height &&
      box.y + box.height > p.y;
    return overlaps && cy > p.y + p.height * 0.52;
  });
}

export const Promo = ({ poster, regions, characters, cutouts }) => {
  const { durationInFrames: dur, width, height } = useVideoConfig();
  const frame = useCurrentFrame();
  const wave = useLoopWave();
  const posterSrc = assetSrc(poster);
  const allRegions = Array.isArray(regions) ? regions : [];
  const allChars = Array.isArray(characters) ? characters : [];
  const allCutouts = Array.isArray(cutouts) ? cutouts : [];

  const people = allRegions.filter(isPersonRegion);
  const ui = allRegions.filter(isUiRegion);
  const plaqueCutouts = allCutouts.filter(isPlaqueCutout);
  const plaqueBack = plaqueCutouts.filter((item) => !isFrontCutout(item, allChars));
  const plaqueFront = plaqueCutouts.filter((item) => isFrontCutout(item, allChars));
  const propCutouts = allCutouts.filter((item) => !isPlaqueCutout(item));

  return (
    <AbsoluteFill style={{ background: "#000", overflow: "hidden" }}>
      {posterSrc ? (
        <Img src={posterSrc} style={{ width, height, objectFit: "fill" }} />
      ) : null}

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

      {plaqueBack.map((item) => (
        <CutoutLayer
          key={`cutout-${item.index}`}
          cutout={item}
          canvasW={width}
          canvasH={height}
          frame={frame}
          dur={dur}
          wave={wave}
        />
      ))}

      {allChars.length > 0
        ? allChars.map((char) => (
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
        : people.map((region, idx) => (
            <OverlayFX
              key={`person-${region.key || idx}`}
              region={region}
              canvasW={width}
              canvasH={height}
              dur={dur}
              wave={wave}
            />
          ))}

      {plaqueFront.map((item) => (
        <CutoutLayer
          key={`cutout-front-${item.index}`}
          cutout={item}
          canvasW={width}
          canvasH={height}
          frame={frame}
          dur={dur}
          wave={wave}
        />
      ))}

      {propCutouts.map((item) => (
        <CutoutLayer
          key={`prop-${item.index}`}
          cutout={item}
          canvasW={width}
          canvasH={height}
          frame={frame}
          dur={dur}
          wave={wave}
        />
      ))}
    </AbsoluteFill>
  );
};
