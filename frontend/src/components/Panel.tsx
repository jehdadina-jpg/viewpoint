import type { CSSProperties, ReactNode } from "react";

interface Props {
  title: string;
  sub?: ReactNode;
  tools?: ReactNode;
  children: ReactNode;
  pad?: boolean;
  style?: CSSProperties;
  bodyStyle?: CSSProperties;
  className?: string;
}

/** Terminal panel: amber uppercase title bar + scrollable body. */
export function Panel({ title, sub, tools, children, pad, style, bodyStyle, className }: Props) {
  return (
    <section className={`panel ${className ?? ""}`} style={style}>
      <header className="panel-head">
        <span>
          {title}
          {sub && <span className="sub"> {sub}</span>}
        </span>
        {tools && <span className="tools">{tools}</span>}
      </header>
      <div className={`panel-body ${pad ? "pad" : ""}`} style={bodyStyle}>{children}</div>
    </section>
  );
}
