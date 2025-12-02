import { Coordinates, TriHexChessWrapper } from "tri-hex-chess";
import { NewPiece } from "./class/NewPiece.ts";
import { RefObject, useLayoutEffect, useState } from "react";
import useResizeObserver from "@react-hook/resize-observer";

export const get3PointsCircle = (radius: number) => {
  const points = [];

  for (let angle = 0; angle < Math.PI * 2; angle += (Math.PI * 2) / 3) {
    const x = radius * Math.cos(angle + Math.PI / 2);
    const y = radius * Math.sin(-angle + Math.PI / 2);

    points.push([-x, y]);
  }

  return points;
};

export type LastMove = {
  from: Coordinates;
  to: Coordinates;
};

export const exportSVG = async (element: SVGSVGElement) => {
  const svgClone = element.cloneNode(true) as SVGElement;

  svgClone.querySelectorAll(".dont-export-svg").forEach((el) => el.remove());

  const imageElements = svgClone.querySelectorAll("image");

  // fetch and replace images dynamically while preserving attributes
  const fetchPromises = Array.from(imageElements).map(async (imageEl) => {
    const href = imageEl.getAttribute("href") || imageEl.getAttribute("xlink:href");
    const width = imageEl.getAttribute("width");
    const height = imageEl.getAttribute("height");
    const transform = imageEl.getAttribute("transform") || "";
    const opacity = imageEl.getAttribute("opacity") || "1";

    // this can error out, but it shouldn't if the network doesn't fail
    if (href) {
      const response = await fetch(href);
      const text = await response.text();
      const parser = new DOMParser();
      const inlineSVG = parser.parseFromString(text, "image/svg+xml").documentElement;

      // apply original size as a wrapper group
      const wrapper = document.createElementNS("http://www.w3.org/2000/svg", "g");
      wrapper.setAttribute("transform", transform);
      wrapper.setAttribute("opacity", opacity);
      inlineSVG.setAttribute("width", width || "");
      inlineSVG.setAttribute("height", height || "");

      wrapper.appendChild(inlineSVG);

      // replace <image> with the new <g> wrapped SVG
      imageEl.replaceWith(wrapper);
    }
  });

  await Promise.all(fetchPromises);

  const serializer = new XMLSerializer();
  const svgString = serializer.serializeToString(svgClone);
  const blob = new Blob([svgString], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "exported.svg";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

export const rotateArray = <T>(arr: T[], n: number): T[] => {
  return arr.slice(n).concat(arr.slice(0, n));
};

export const getPieces = (game: TriHexChessWrapper, rotation: number): NewPiece[] => {
  const pieces: NewPiece[] = [];

  for (const piece of game.getPieces()) {
    pieces.push(new NewPiece({ type: "ref", ref: piece, rotation }));
  }

  return pieces;
};

export const useSize = (target: RefObject<HTMLElement | SVGSVGElement>) => {
  const [size, setSize] = useState<DOMRect | null>(null);

  useLayoutEffect(() => {
    if (!target.current) return;

    setSize(target.current.getBoundingClientRect());
  }, [target]);

  // Where the magic happens
  useResizeObserver(target, (entry) => setSize(entry.contentRect));
  return size;
};
