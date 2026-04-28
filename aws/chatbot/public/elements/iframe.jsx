// Inline iframe for embedding XANES plots and crystal-structure viewers
// inside the Chainlit chat. `props.src` is a URL served by the FastAPI
// staticfiles mount (e.g. /plots/mp-2657_xanes.html).

export default function Iframe({ src, height = 520, title = "" }) {
  return (
    <div style={{ width: "100%", marginTop: 6 }}>
      <iframe
        src={src}
        title={title}
        style={{
          width: "100%",
          height: `${height}px`,
          border: "1px solid #2a2a2a",
          borderRadius: 8,
          background: "#fff"
        }}
        sandbox="allow-scripts allow-same-origin allow-popups"
        loading="lazy"
      />
    </div>
  );
}
