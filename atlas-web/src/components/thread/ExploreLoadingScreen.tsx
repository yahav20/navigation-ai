"use client";

const CSS = `
  .els-ui {
    position: absolute;
    bottom: 28%;
    left: 50%;
    transform: translateX(-50%);
    text-align: center;
    white-space: nowrap;
    opacity: 0;
    animation: els-reveal 0.9s 1s cubic-bezier(0.33,1,0.68,1) forwards;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', system-ui, sans-serif;
  }
  @keyframes els-reveal {
    from { opacity: 0; transform: translateX(-50%) translateY(8px); }
    to   { opacity: 1; transform: translateX(-50%) translateY(0); }
  }
  .els-primary {
    font-size: 14px;
    color: var(--muted-foreground);
    font-weight: 500;
    letter-spacing: 0.02em;
    margin-bottom: 9px;
  }
  .els-status-wrap { position: relative; height: 16px; margin-bottom: 18px; }
  .els-msg {
    position: absolute;
    left: 50%;
    top: 0;
    transform: translateX(-50%);
    font-size: 10px;
    color: var(--muted-foreground);
    opacity: 0;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    font-weight: 400;
  }
  .els-s1 { animation: els-cycle 10s 1.2s  infinite; }
  .els-s2 { animation: els-cycle 10s 3.7s  infinite; }
  .els-s3 { animation: els-cycle 10s 6.2s  infinite; }
  .els-s4 { animation: els-cycle 10s 8.7s  infinite; }
  @keyframes els-cycle {
    0%   { opacity: 0; transform: translateX(-50%) translateY(5px); }
    8%   { opacity: 0.6; transform: translateX(-50%) translateY(0); }
    19%  { opacity: 0.6; transform: translateX(-50%) translateY(0); }
    25%  { opacity: 0; transform: translateX(-50%) translateY(-5px); }
    100% { opacity: 0; }
  }
  .els-waypoints { display: flex; align-items: center; justify-content: center; }
  .els-wp {
    width: 7px; height: 7px;
    border-radius: 50%;
    border: 1.5px solid rgba(255,255,255,0.22);
    background: transparent;
    flex-shrink: 0;
  }
  .els-wp-dash { width: 20px; height: 1px; background: rgba(255,255,255,0.10); flex-shrink: 0; }
  .els-wp1 { animation: els-glow 2.1s 0s   ease-in-out infinite; }
  .els-wp2 { animation: els-glow 2.1s 0.7s ease-in-out infinite; }
  .els-wp3 { animation: els-glow 2.1s 1.4s ease-in-out infinite; }
  @keyframes els-glow {
    0%,100% { border-color: rgba(255,255,255,0.18); box-shadow: none; background: transparent; }
    50%     { border-color: rgba(255,255,255,0.88);
              box-shadow: 0 0 0 2px rgba(255,255,255,0.06), 0 0 10px 2px rgba(255,255,255,0.14);
              background: rgba(255,255,255,0.08); }
  }
`;

export default function ExploreLoadingScreen() {
  return (
    <div style={{ position: "relative", width: "100%", height: "420px", overflow: "hidden" }}>
      {/* eslint-disable-next-line react/no-danger */}
      <style dangerouslySetInnerHTML={{ __html: CSS }} />

      {/* ── SVG scene ─────────────────────────────────────── */}
      <svg
        viewBox="0 0 1200 700"
        preserveAspectRatio="xMidYMid slice"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <path id="els-fp" d="M -95,572 C 80,505 270,335 595,258 C 845,196 1060,188 1315,150" />
          <filter id="els-mk" x="-120%" y="-120%" width="340%" height="340%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="2.2" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* World grid */}
        <g>
          <animate attributeName="opacity" from="0" to="0.055" dur="1.2s" begin="0.8s" fill="freeze" />
          <path d="M 200,-8 Q 185,348 212,708"   stroke="white" strokeWidth="0.65" fill="none" />
          <path d="M 400,-8 Q 387,348 404,708"   stroke="white" strokeWidth="0.65" fill="none" />
          <path d="M 600,-8 Q 594,348 598,708"   stroke="white" strokeWidth="0.65" fill="none" />
          <path d="M 800,-8 Q 797,348 796,708"   stroke="white" strokeWidth="0.65" fill="none" />
          <path d="M 1000,-8 Q 1003,348 999,708" stroke="white" strokeWidth="0.65" fill="none" />
          <path d="M -8,175 Q 600,162 1208,175"  stroke="white" strokeWidth="0.65" fill="none" />
          <path d="M -8,350 Q 600,337 1208,350"  stroke="white" strokeWidth="0.65" fill="none" />
          <path d="M -8,525 Q 600,512 1208,525"  stroke="white" strokeWidth="0.65" fill="none" />
        </g>

        {/* Altitude streams */}
        <g opacity="0.032">
          <path d="M -250,188 C 150,180 620,196 1040,184 S 1500,176 1850,190" stroke="white" strokeWidth="1.2" fill="none">
            <animateTransform attributeName="transform" type="translate" from="0,0" to="-320,0" dur="19s" repeatCount="indefinite" />
          </path>
          <path d="M -180,294 C 220,284 680,300 1100,288 S 1560,280 1880,296" stroke="white" strokeWidth="0.9" fill="none">
            <animateTransform attributeName="transform" type="translate" from="0,0" to="-320,0" dur="25s" repeatCount="indefinite" />
          </path>
          <path d="M -120,398 C 320,387 780,404 1200,392 S 1640,383 1960,400" stroke="white" strokeWidth="1.0" fill="none">
            <animateTransform attributeName="transform" type="translate" from="0,0" to="-320,0" dur="21s" repeatCount="indefinite" />
          </path>
        </g>

        {/* Background clouds (0.06, 32 s) */}
        <g>
          <animate attributeName="opacity" from="0" to="0.06" dur="1.8s" begin="0.5s" fill="freeze" />
          <animateTransform attributeName="transform" type="translate" from="0,0" to="-1200,0" dur="32s" repeatCount="indefinite" />
          <ellipse cx="110"  cy="208" rx="195" ry="27" fill="white" />
          <ellipse cx="182"  cy="200" rx="122" ry="19" fill="white" />
          <ellipse cx="575"  cy="348" rx="225" ry="29" fill="white" />
          <ellipse cx="652"  cy="339" rx="140" ry="21" fill="white" />
          <ellipse cx="1025" cy="165" rx="208" ry="25" fill="white" />
          <ellipse cx="1310" cy="208" rx="195" ry="27" fill="white" />
          <ellipse cx="1382" cy="200" rx="122" ry="19" fill="white" />
          <ellipse cx="1775" cy="348" rx="225" ry="29" fill="white" />
          <ellipse cx="1852" cy="339" rx="140" ry="21" fill="white" />
          <ellipse cx="2225" cy="165" rx="208" ry="25" fill="white" />
        </g>

        {/* Mid clouds (0.10, 22 s) */}
        <g>
          <animate attributeName="opacity" from="0" to="0.10" dur="1.8s" begin="0.5s" fill="freeze" />
          <animateTransform attributeName="transform" type="translate" from="0,0" to="-1200,0" dur="22s" repeatCount="indefinite" />
          <ellipse cx="318"  cy="458" rx="172" ry="21" fill="white" />
          <ellipse cx="388"  cy="450" rx="108" ry="15" fill="white" />
          <ellipse cx="808"  cy="118" rx="185" ry="23" fill="white" />
          <ellipse cx="1182" cy="518" rx="157" ry="19" fill="white" />
          <ellipse cx="1518" cy="458" rx="172" ry="21" fill="white" />
          <ellipse cx="1588" cy="450" rx="108" ry="15" fill="white" />
          <ellipse cx="2008" cy="118" rx="185" ry="23" fill="white" />
          <ellipse cx="2382" cy="518" rx="157" ry="19" fill="white" />
        </g>

        {/* Foreground clouds (0.14, 14 s) */}
        <g>
          <animate attributeName="opacity" from="0" to="0.14" dur="1.8s" begin="0.5s" fill="freeze" />
          <animateTransform attributeName="transform" type="translate" from="0,0" to="-1200,0" dur="14s" repeatCount="indefinite" />
          <ellipse cx="472"  cy="574" rx="152" ry="18" fill="white" />
          <ellipse cx="537"  cy="566" rx="92"  ry="13" fill="white" />
          <ellipse cx="972"  cy="638" rx="168" ry="20" fill="white" />
          <ellipse cx="1448" cy="558" rx="142" ry="17" fill="white" />
          <ellipse cx="1672" cy="574" rx="152" ry="18" fill="white" />
          <ellipse cx="1737" cy="566" rx="92"  ry="13" fill="white" />
          <ellipse cx="2172" cy="638" rx="168" ry="20" fill="white" />
          <ellipse cx="2648" cy="558" rx="142" ry="17" fill="white" />
        </g>

        {/* Flight trail */}
        <path
          d="M -95,572 C 80,505 270,335 595,258 C 845,196 1060,188 1315,150"
          stroke="rgba(255,255,255,0.38)"
          strokeWidth="1.5"
          strokeDasharray="5,9"
          fill="none"
          strokeLinecap="round"
        >
          <animate attributeName="strokeDashoffset" values="1000;0" dur="5s" repeatCount="indefinite" calcMode="linear" />
          <animate attributeName="opacity" values="0;0;0.42;0.42;0" keyTimes="0;0.04;0.13;0.87;1" dur="5s" repeatCount="indefinite" />
        </path>

        {/* Destination markers */}
        <g>
          <animate attributeName="opacity" from="0" to="1" dur="0.9s" begin="0.8s" fill="freeze" />
          <line x1="272" y1="320" x2="542" y2="260" stroke="rgba(255,255,255,0.075)" strokeWidth="0.8" strokeDasharray="4,9" />
          <line x1="542" y1="260" x2="808" y2="244" stroke="rgba(255,255,255,0.075)" strokeWidth="0.8" strokeDasharray="4,9" />

          <g transform="translate(272,320)">
            <circle r="8" fill="none" stroke="rgba(255,255,255,0.14)" strokeWidth="0.8">
              <animate attributeName="r"       values="7;14;7"      dur="3s"   repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.35;0;0.35"  dur="3s"   repeatCount="indefinite" />
            </circle>
            <circle r="3" fill="white" filter="url(#els-mk)">
              <animate attributeName="opacity" values="0.42;1;0.42" dur="3s"   repeatCount="indefinite" />
            </circle>
          </g>

          <g transform="translate(542,260)">
            <circle r="8" fill="none" stroke="rgba(255,255,255,0.14)" strokeWidth="0.8">
              <animate attributeName="r"       values="7;14;7"      dur="3.5s" begin="1.1s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.35;0;0.35"  dur="3.5s" begin="1.1s" repeatCount="indefinite" />
            </circle>
            <circle r="3" fill="white" filter="url(#els-mk)">
              <animate attributeName="opacity" values="0.42;1;0.42" dur="3.5s" begin="1.1s" repeatCount="indefinite" />
            </circle>
          </g>

          <g transform="translate(808,244)">
            <circle r="8" fill="none" stroke="rgba(255,255,255,0.14)" strokeWidth="0.8">
              <animate attributeName="r"       values="7;14;7"      dur="2.8s" begin="2.3s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="0.35;0;0.35"  dur="2.8s" begin="2.3s" repeatCount="indefinite" />
            </circle>
            <circle r="3" fill="white" filter="url(#els-mk)">
              <animate attributeName="opacity" values="0.42;1;0.42" dur="2.8s" begin="2.3s" repeatCount="indefinite" />
            </circle>
          </g>
        </g>

        {/* Stars */}
        <circle cx="132" cy="70"   r="1"   fill="white"><animate attributeName="opacity" values="0.15;0.82;0.15" dur="3.4s"            repeatCount="indefinite" /></circle>
        <circle cx="388" cy="42"   r="0.8" fill="white"><animate attributeName="opacity" values="0.22;0.90;0.22" dur="4.3s" begin="1.3s" repeatCount="indefinite" /></circle>
        <circle cx="674" cy="78"   r="1"   fill="white"><animate attributeName="opacity" values="0.10;0.76;0.10" dur="2.9s" begin="0.7s" repeatCount="indefinite" /></circle>
        <circle cx="918" cy="50"   r="0.8" fill="white"><animate attributeName="opacity" values="0.18;0.86;0.18" dur="4.0s" begin="2.2s" repeatCount="indefinite" /></circle>
        <circle cx="1090" cy="88"  r="1"   fill="white"><animate attributeName="opacity" values="0.25;0.80;0.25" dur="4.7s" begin="1.1s" repeatCount="indefinite" /></circle>
        <circle cx="52"  cy="130"  r="0.8" fill="white"><animate attributeName="opacity" values="0.12;0.72;0.12" dur="3.8s" begin="0.4s" repeatCount="indefinite" /></circle>

        {/* Aircraft — nose points along +x, animateMotion rotate="auto" does the rest */}
        <g>
          <animateMotion dur="5s" repeatCount="indefinite" rotate="auto" calcMode="spline" keyTimes="0;1" keySplines="0.37 0 0.63 1">
            <mpath href="#els-fp" />
          </animateMotion>
          {/* Fuselage */}
          <path d="M -40,0 C -30,-4.8 -6,-5.8 18,-3.4 L 33,-2.1 C 37,-1.1 39,0 37,1.3 L 31,2.3 C -6,5.8 -30,4.8 -40,0 Z" fill="white" />
          {/* Nose */}
          <path d="M 31,-2.1 C 41,-1.3 48,0 41,2.1 L 31,2.3 Z" fill="white" />
          {/* Port wing */}
          <path d="M 7,-3.2 C 4,-14 -5,-29 -9,-35 L -15,-32 C -11,-26 -2,-12 0,-2.2 Z" fill="rgba(255,255,255,0.90)" />
          {/* Starboard wing */}
          <path d="M 7,3.2 C 4,14 -5,29 -9,35 L -15,32 C -11,26 -2,12 0,2.2 Z" fill="rgba(255,255,255,0.90)" />
          {/* Port stabiliser */}
          <path d="M -28,-2.2 C -31,-9 -35,-18 -36,-18 L -32,-17 C -32,-16 -29,-9 -26,-2.2 Z" fill="rgba(255,255,255,0.80)" />
          {/* Starboard stabiliser */}
          <path d="M -28,2.2 C -31,9 -35,18 -36,18 L -32,17 C -32,16 -29,9 -26,2.2 Z" fill="rgba(255,255,255,0.80)" />
          {/* Vertical fin */}
          <path d="M -31,-1.2 C -33,-7 -36,-17 -34,-19 L -30,-16 C -30,-14 -29,-7 -27,-1.2 Z" fill="rgba(255,255,255,0.76)" />
          {/* Engines */}
          <ellipse cx="-2" cy="-23" rx="8.5" ry="2.8" fill="rgba(255,255,255,0.74)" />
          <ellipse cx="-2" cy="23"  rx="8.5" ry="2.8" fill="rgba(255,255,255,0.74)" />
        </g>
      </svg>

      {/* ── Loading text overlay ───────────────────────────── */}
      <div className="els-ui">
        <p className="els-primary">Finding your perfect destination</p>
        <div className="els-status-wrap">
          <span className="els-msg els-s1">Scanning hidden gems</span>
          <span className="els-msg els-s2">Analyzing flight routes</span>
          <span className="els-msg els-s3">Matching your travel style</span>
          <span className="els-msg els-s4">Discovering unforgettable places</span>
        </div>
        <div className="els-waypoints">
          <div className="els-wp els-wp1" />
          <div className="els-wp-dash" />
          <div className="els-wp els-wp2" />
          <div className="els-wp-dash" />
          <div className="els-wp els-wp3" />
        </div>
      </div>
    </div>
  );
}
