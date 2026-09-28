import React from "react";

type Props = { size?: number; strokeWidth?: number };

const Svg = ({ children, size = 20, strokeWidth = 1.8 }: Props & { children: React.ReactNode }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {children}
  </svg>
);

export function SparklesIcon(p: Props) { return <Svg {...p}><path d="m12 3-1.1 4.2L7 9l3.9 1.8L12 15l1.1-4.2L17 9l-3.9-1.8L12 3Z"/><path d="m19 14-.7 2.3L16 17l2.3.7L19 20l.7-2.3L22 17l-2.3-.7L19 14Z"/></Svg>; }
export function PlusIcon(p: Props) { return <Svg {...p}><path d="M12 5v14M5 12h14"/></Svg>; }
export function SendIcon(p: Props) { return <Svg {...p}><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></Svg>; }
export function UserIcon(p: Props) { return <Svg {...p}><circle cx="12" cy="8" r="3.4"/><path d="M5 21c.7-4 3-6 7-6s6.3 2 7 6"/></Svg>; }
export function CheckIcon(p: Props) { return <Svg {...p}><path d="m5 12 4 4L19 6"/></Svg>; }
export function BookIcon(p: Props) { return <Svg {...p}><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H12v17H6.5A2.5 2.5 0 0 0 4 22V5.5Z"/><path d="M20 5.5A2.5 2.5 0 0 0 17.5 3H12v17h5.5A2.5 2.5 0 0 1 20 22V5.5Z"/></Svg>; }
export function ShieldIcon(p: Props) { return <Svg {...p}><path d="M12 3 20 6v6c0 5-3.2 8.3-8 9-4.8-.7-8-4-8-9V6l8-3Z"/><path d="m8.5 12 2.2 2.2 4.8-5"/></Svg>; }
export function LinkIcon(p: Props) { return <Svg {...p}><path d="M10 13a5 5 0 0 0 7.1.1l2-2a5 5 0 0 0-7.1-7.1l-1.1 1.1"/><path d="M14 11a5 5 0 0 0-7.1-.1l-2 2A5 5 0 0 0 7 20l1.1-1.1"/></Svg>; }
export function ChevronDownIcon(p: Props) { return <Svg {...p}><path d="m6 9 6 6 6-6"/></Svg>; }
export function XIcon(p: Props) { return <Svg {...p}><path d="m6 6 12 12M18 6 6 18"/></Svg>; }
export function InfoIcon(p: Props) { return <Svg {...p}><circle cx="12" cy="12" r="9"/><path d="M12 10v6M12 7h.01"/></Svg>; }
export function MenuIcon(p: Props) { return <Svg {...p}><path d="M4 6h16M4 12h16M4 18h16"/></Svg>; }
export function WarningIcon(p: Props) { return <Svg {...p}><path d="M12 3 22 20H2L12 3Z"/><path d="M12 9v5M12 17h.01"/></Svg>; }
