# AutoArchitect AI — Design System
> Canonical reference. Derived from Linear's exact design language.

---

## Color Tokens

### Surfaces — 4-step ladder (darkest to lightest)
| Token | Value | Use |
|-------|-------|-----|
| `--canvas` | `#010102` | Page background |
| `--surface-1` | `#1c1c1f` | Cards, panels |
| `--surface-2` | `#232328` | Hovered cards, inputs |
| `--surface-3` | `#2a2a30` | Dropdowns, sub-nav |
| `--surface-4` | `#303038` | Deepest lifted elements |

### Hairlines — borders, dividers
| Token | Value | Use |
|-------|-------|-----|
| `--hairline` | `#23252a` | Default borders |
| `--hairline-strong` | `#3a3d44` | Emphasized borders, hover |
| `--hairline-tertiary` | `#1a1c20` | Subtlest dividers |

### Brand — lavender, used scarcely
| Token | Value | Use |
|-------|-------|-----|
| `--primary` | `#5e6ad2` | Primary CTA, focus ring |
| `--primary-hover` | `#828fff` | Hover state, accent text |
| `--primary-focus` | `#5e69d1` | Focus ring |

> **Rule:** Lavender appears ONLY on primary CTA buttons, brand mark, focus rings, and active states. Nowhere else.

### Text — ink ladder
| Token | Value | Use |
|-------|-------|-----|
| `--ink` | `#f7f8f8` | Primary text, headings |
| `--ink-muted` | `#d0d6e0` | Secondary text, body |
| `--ink-subtle` | `#8a8f98` | Tertiary text, labels |
| `--ink-tertiary` | `#62666d` | Disabled, hints |

### Semantic — one green only
| Token | Value | Use |
|-------|-------|-----|
| `--success` | `#27a644` | Success border |
| `--success-h` | `#34d058` | Success text, icons |
| `--success-dim` | `rgba(39,166,68,0.10)` | Success background tint |
| `--warn` | `#d97706` | Warning text |
| `--danger` | `#dc2626` | Error text |

---

## Spacing — 4px base grid
| Token | Value |
|-------|-------|
| `--space-xxs` | `4px` |
| `--space-xs` | `8px` |
| `--space-sm` | `12px` |
| `--space-md` | `16px` |
| `--space-lg` | `24px` |
| `--space-xl` | `32px` |
| `--space-xxl` | `48px` |
| `--space-section` | `96px` |

---

## Border Radius — restrained
| Token | Value | Use |
|-------|-------|-----|
| `--r-xs` | `4px` | Inline elements |
| `--r-sm` | `6px` | Chips, small controls |
| `--r-md` | `8px` | Buttons, inputs (default) |
| `--r-lg` | `12px` | Cards, panels |
| `--r-xl` | `16px` | Large containers |
| `--r-pill` | `9999px` | Status pills, badges ONLY |

---

## Typography

### Font stack
```css
font-family: -apple-system, "SF Pro Display", BlinkMacSystemFont,
             "Inter", system-ui, "Segoe UI", Roboto, sans-serif;
```

### Type scale
| Role | Size | Weight | Tracking | Line-height |
|------|------|--------|----------|-------------|
| Hero / display-xl | 80px → 40px mobile | 600 | -3.0px | 1.05 |
| Section headline / display-md | 40px | 600 | -1.0px | 1.15 |
| Card title | 22px | 500 | -0.4px | 1.25 |
| Body | 16px | 400 | -0.05px | 1.5 |
| Eyebrow | 13px | 500 | +0.4px | — |
| Small / meta | 13px | 400–500 | 0 | 1.4 |
| Hint / label | 11–12px | 400–500 | 0 | — |

> **Rule:** Display text uses aggressive negative tracking (-3px). Eyebrow text uses positive tracking (+0.4px) for contrast.

---

## Components

### Buttons
```css
/* Primary — solid lavender, NO gradient */
.btn-primary {
  background:    var(--primary);   /* #5e6ad2 */
  color:         #fff;
  font-size:     14px;
  font-weight:   500;
  padding:       8px 14px;
  border-radius: var(--r-md);      /* 8px */
  border:        none;
  transition:    background 150ms ease;
}
.btn-primary:hover { background: var(--primary-hover); }

/* Secondary — surface-1 with hairline */
.btn-secondary {
  background:    var(--surface-1);
  color:         var(--ink);
  border:        1px solid var(--hairline);
  font-size:     14px;
  font-weight:   500;
  padding:       8px 14px;
  border-radius: var(--r-md);
}
.btn-secondary:hover {
  background:   var(--surface-2);
  border-color: var(--hairline-strong);
}
```

### Cards
```css
.card {
  background:    var(--surface-1);
  border:        1px solid var(--hairline);
  border-radius: var(--r-lg);   /* 12px */
  padding:       var(--space-lg); /* 24px */
  /* NO backdrop-filter, NO box-shadow, NO gradients */
}
.card:hover {
  background:   var(--surface-2);        /* surface lift */
  border-color: var(--hairline-strong);
}
```

---

## What NOT to do

| Anti-pattern | Why |
|---|---|
| `backdrop-filter: blur(20px)` | Not Linear — that's glassmorphism |
| `radial-gradient` mesh backgrounds | Linear uses solid `--canvas` |
| `border-radius: 20px` on cards | Too large. Linear uses 12px max |
| Gradient buttons (indigo→purple) | Solid lavender only |
| `inset 0 1px 0 rgba(255,255,255,0.05)` highlights | Use hairline border only |
| Box-shadow glow on cards | Surface lift (background color change) only |
| Lavender on secondary text, decorative elements | Scarcely used — CTA + brand only |
