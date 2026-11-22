import React, { useEffect, useRef, useState, Component } from 'react';
import * as THREE from 'three';

// Global flag to remember WebGL failure across renders and prevent repeated retry loops
let hasWebGLFailedEver = false;

/**
 * Safe check to verify whether WebGL context creation is supported
 * without throwing unhandled exceptions or crashing the React application.
 */
export function isWebGLAvailable() {
  if (hasWebGLFailedEver) return false;
  if (typeof window === 'undefined' || typeof document === 'undefined') return false;

  try {
    if (!window.WebGLRenderingContext) {
      hasWebGLFailedEver = true;
      return false;
    }
    const canvas = document.createElement('canvas');
    const gl =
      canvas.getContext('webgl2') ||
      canvas.getContext('webgl') ||
      canvas.getContext('experimental-webgl');

    if (!gl) {
      hasWebGLFailedEver = true;
      return false;
    }

    if (typeof gl.isContextLost === 'function' && gl.isContextLost()) {
      hasWebGLFailedEver = true;
      return false;
    }

    return true;
  } catch (e) {
    hasWebGLFailedEver = true;
    return false;
  }
}

/**
 * Beautiful, lightweight CSS-based ambient background.
 * Zero WebGL, zero canvas, zero GPU memory overhead.
 * Matches the Flora-inspired luminous aesthetic seamlessly.
 */
export function CssHeroBackground() {
  return (
    <div
      className="absolute inset-0 pointer-events-none overflow-hidden opacity-60 z-0"
      aria-hidden="true"
    >
      {/* Ambient luminous glow 1 - Emerald & Sky */}
      <div
        className="absolute -top-24 -left-20 w-96 h-48 rounded-full blur-3xl opacity-35 animate-pulse"
        style={{
          background: 'radial-gradient(ellipse at center, rgba(16, 185, 129, 0.5) 0%, rgba(56, 189, 248, 0.25) 50%, transparent 75%)',
          animationDuration: '6s',
        }}
      />
      {/* Ambient luminous glow 2 - Indigo & Violet */}
      <div
        className="absolute -top-20 -right-20 w-96 h-48 rounded-full blur-3xl opacity-30"
        style={{
          background: 'radial-gradient(ellipse at center, rgba(99, 102, 241, 0.45) 0%, rgba(56, 189, 248, 0.2) 60%, transparent 80%)',
        }}
      />
      {/* Technical micro-dot grid overlay */}
      <div
        className="absolute inset-0 opacity-[0.035]"
        style={{
          backgroundImage:
            'radial-gradient(rgba(255, 255, 255, 0.6) 1px, transparent 1px)',
          backgroundSize: '24px 24px',
        }}
      />
    </div>
  );
}

/**
 * Error boundary component to guarantee that Three.js or WebGL errors
 * NEVER crash the React application tree.
 */
export class CanvasErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    hasWebGLFailedEver = true;
    return { hasError: true };
  }

  componentDidCatch(error) {
    hasWebGLFailedEver = true;
    console.warn(
      'CanvasErrorBoundary caught an error in Three.js hero canvas. Rendering CSS fallback.',
      error?.message || error
    );
  }

  render() {
    if (this.state.hasError) {
      return <CssHeroBackground />;
    }
    return this.props.children;
  }
}

/**
 * Inner Three.js canvas component with safe context initialization,
 * comprehensive try/catch guards, and complete lifecycle cleanup.
 */
function ThreeHeroCanvasInner() {
  const mountRef = useRef(null);
  const [webGLAvailable, setWebGLAvailable] = useState(() => isWebGLAvailable());

  useEffect(() => {
    // If WebGL is unavailable or has failed, abort without touching Three.js
    if (!webGLAvailable || hasWebGLFailedEver) return;

    const mount = mountRef.current;
    if (!mount) return;

    let renderer = null;
    let scene = null;
    let camera = null;
    let geometry = null;
    let material = null;
    let particles = null;
    let animId = null;
    let clock = null;
    let isDisposed = false;

    let handleMouseMove = null;
    let handleResize = null;
    let handleContextLost = null;

    try {
      const width = mount.clientWidth || 300;
      const height = mount.clientHeight || 80;

      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(60, width / Math.max(height, 1), 0.1, 1000);
      camera.position.z = 80;

      // Safe creation of WebGLRenderer with power-conscious settings
      renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true,
        powerPreference: 'low-power',
        failIfMajorPerformanceCaveat: false,
      });

      if (!renderer || !renderer.domElement) {
        throw new Error('Renderer DOM element unavailable');
      }

      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      mount.appendChild(renderer.domElement);

      // Handle context lost event gracefully during runtime
      handleContextLost = (e) => {
        if (e && e.preventDefault) e.preventDefault();
        console.warn('WebGL context lost. Gracefully falling back to CSS ambient background.');
        hasWebGLFailedEver = true;
        if (!isDisposed) {
          setWebGLAvailable(false);
        }
      };
      renderer.domElement.addEventListener('webglcontextlost', handleContextLost, false);

      // Particle Geometry: Ambient grid/wave
      const count = 350;
      const positions = new Float32Array(count * 3);
      const colors = new Float32Array(count * 3);

      const c1 = new THREE.Color('#10b981'); // Emerald
      const c2 = new THREE.Color('#38bdf8'); // Sky
      const c3 = new THREE.Color('#6366f1'); // Indigo

      for (let i = 0; i < count; i++) {
        const i3 = i * 3;
        positions[i3] = (Math.random() - 0.5) * 160;
        positions[i3 + 1] = (Math.random() - 0.5) * 60;
        positions[i3 + 2] = (Math.random() - 0.5) * 60;

        const mixed = Math.random();
        const col = mixed < 0.5 ? c1.clone().lerp(c2, mixed * 2) : c2.clone().lerp(c3, (mixed - 0.5) * 2);
        colors[i3] = col.r;
        colors[i3 + 1] = col.g;
        colors[i3 + 2] = col.b;
      }

      geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

      material = new THREE.PointsMaterial({
        size: 2.2,
        vertexColors: true,
        transparent: true,
        opacity: 0.65,
        blending: THREE.AdditiveBlending,
      });

      particles = new THREE.Points(geometry, material);
      scene.add(particles);

      // Mouse reactivity
      let mouseX = 0;
      let mouseY = 0;
      let targetX = 0;
      let targetY = 0;

      handleMouseMove = (e) => {
        mouseX = (e.clientX / window.innerWidth) * 2 - 1;
        mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
      };
      window.addEventListener('mousemove', handleMouseMove, { passive: true });

      // Resize handler
      handleResize = () => {
        if (!mount || !renderer || !camera || isDisposed) return;
        const w = mount.clientWidth;
        const h = mount.clientHeight;
        if (w === 0 || h === 0) return;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      };
      window.addEventListener('resize', handleResize);

      // Animation Loop
      clock = new THREE.Clock();

      const animate = () => {
        if (isDisposed) return;
        animId = requestAnimationFrame(animate);

        try {
          const elapsed = clock.getElapsedTime();

          targetX += (mouseX * 12 - targetX) * 0.05;
          targetY += (mouseY * 8 - targetY) * 0.05;
          particles.rotation.y = elapsed * 0.04 + targetX * 0.02;
          particles.rotation.x = Math.sin(elapsed * 0.03) * 0.1 + targetY * 0.02;

          const pos = geometry.attributes.position.array;
          for (let i = 0; i < count; i++) {
            const i3 = i * 3;
            const x = pos[i3];
            pos[i3 + 1] += Math.sin(elapsed * 1.5 + x * 0.05) * 0.04;
          }
          geometry.attributes.position.needsUpdate = true;

          renderer.render(scene, camera);
        } catch (renderError) {
          console.warn('ThreeHeroCanvas animation error:', renderError?.message || renderError);
          cancelAnimationFrame(animId);
          hasWebGLFailedEver = true;
          setWebGLAvailable(false);
        }
      };

      animate();
    } catch (err) {
      console.warn('WebGL context creation failed or unavailable. Falling back to CSS ambient background:', err?.message || err);
      hasWebGLFailedEver = true;
      setWebGLAvailable(false);
    }

    // Complete, safe cleanup of resources, listeners, and renderer
    return () => {
      isDisposed = true;
      if (animId) {
        cancelAnimationFrame(animId);
        animId = null;
      }
      if (handleMouseMove) {
        window.removeEventListener('mousemove', handleMouseMove);
      }
      if (handleResize) {
        window.removeEventListener('resize', handleResize);
      }
      if (renderer && renderer.domElement && handleContextLost) {
        try {
          renderer.domElement.removeEventListener('webglcontextlost', handleContextLost);
        } catch (e) {}
      }
      if (mount && renderer && renderer.domElement) {
        try {
          if (mount.contains(renderer.domElement)) {
            mount.removeChild(renderer.domElement);
          }
        } catch (e) {}
      }
      if (geometry) {
        try { geometry.dispose(); } catch (e) {}
      }
      if (material) {
        try { material.dispose(); } catch (e) {}
      }
      if (renderer) {
        try { renderer.dispose(); } catch (e) {}
      }
    };
  }, [webGLAvailable]);

  // If WebGL is not available, render the beautiful CSS animated background
  if (!webGLAvailable || hasWebGLFailedEver) {
    return <CssHeroBackground />;
  }

  return (
    <div
      ref={mountRef}
      className="absolute inset-0 pointer-events-none overflow-hidden opacity-60 z-0"
      aria-hidden="true"
    />
  );
}

/**
 * Default export protected by CanvasErrorBoundary.
 * Guarantees the application never crashes even if WebGL or Three.js fails completely.
 */
export default function ThreeHeroCanvas() {
  return (
    <CanvasErrorBoundary>
      <ThreeHeroCanvasInner />
    </CanvasErrorBoundary>
  );
}
