import React, { useState, useEffect } from 'react';
import { BrowserRouter } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Sidebar } from './components/layout/Sidebar';
import { TopNav } from './components/layout/TopNav';
import { SessionProvider } from './context/SessionContext';

// Section Imports
import { IntroSection } from './components/sections/IntroSection';
import { BeforeSection } from './components/sections/BeforeSection';
import { AfterSection } from './components/sections/AfterSection';
import { CoaxSection } from './components/sections/CoaxSection';
import { InstructionsSection } from './components/sections/InstructionsSection';
import { LandingSection } from './components/sections/LandingSection';
import { SmoothScroll } from './components/ui/SmoothScroll';

// DPI Reference Images
import dpiCorrect1 from './assets/dpi_correct_1.png';
import dpiCorrect2 from './assets/dpi_correct_2.png';
import dpiIncorrect1 from './assets/dpi_incorrect_1.png';
import dpiIncorrect2 from './assets/dpi_incorrect_2.png';

const PlaceHolder = ({ name }) => (
  <div className="animate-fade-in" style={{ padding: '2rem', textAlign: 'center' }}>
    <h2 style={{ marginBottom: '1rem', fontFamily: 'Poppins, sans-serif' }}>{name} Section</h2>
    <p>This section is under construction.</p>
  </div>
);

// ── Inner app — must be inside SessionProvider so useSession() works ──────────
function AppInner() {
  const [isLanding, setIsLanding] = useState(true);
  const [activeView, setActiveView] = useState('intro');
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
  const scrollRef = React.useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Scroll to top on view change
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
    window.scrollTo(0, 0);
  }, [activeView]);

  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');

  const renderContent = () => {
    switch (activeView) {
      case 'intro': return <IntroSection onStart={() => setActiveView('before')} />;
      case 'before': return <BeforeSection />;
      case 'after': return <AfterSection />;
      case 'coax': return <CoaxSection />;
      case 'instructions': return <InstructionsSection />;
      case 'help': return (
        <div className="section-container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div className="section-header">
            <h2 className="section-title">How to Use Network Mapper</h2>
            <p className="section-subtitle">A brief guide to understanding the workflow and validating configuration.</p>
          </div>
          <div style={{ backgroundColor: 'var(--bg-secondary)', padding: '2rem', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-color)' }}>
            <ol style={{ paddingLeft: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem', color: 'var(--text-secondary)' }}>
              <li><strong>Intro:</strong> Enter a Map Name and upload the Prism screenshot to start a session.</li>
              <li><strong>Before:</strong> Upload your baseline infrastructure map and provide metadata.</li>
              <li><strong>After:</strong> Configure connectivity options — fields entered in Fiber Map auto-fill other fiber tabs.</li>
              <li><strong>Coax/Fiber:</strong> Use specific modules to overlay new updates based on network type.</li>
              <li><strong>Instructions:</strong> Test alert behaviors and visual cues for standard scenarios.</li>
            </ol>
          </div>

          <div className="section-header" style={{ marginTop: '1.5rem', marginBottom: '0.5rem' }}>
            <h2 className="section-title" style={{ fontSize: '1.5rem' }}>DPI Calibration Reference</h2>
            <p className="section-subtitle">
              During the map alignment stage, a preview modal displays sample tiles. 
              <strong> If the preview tiles in the modal look like the Correct DPI images below</strong>, then the selected DPI is correct and you can safely click proceed. 
              <strong> If the preview tiles look like the Incorrect DPI images (showing broken component shapes, text borders cut at edges, or partial circles)</strong>, you must change the DPI setting and re-run.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '2rem' }}>
            {/* Column 1: Correct DPI */}
            <div style={{ 
              display: 'flex', flexDirection: 'column', gap: '1.5rem', padding: '2rem', 
              backgroundColor: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)', 
              border: '2px solid rgba(16, 185, 129, 0.4)', position: 'relative',
              boxShadow: 'var(--shadow-sm)'
            }}>
              <div style={{ 
                position: 'absolute', top: '1.25rem', right: '1.25rem', padding: '0.25rem 0.75rem', borderRadius: 'var(--radius-pill)',
                fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px',
                backgroundColor: 'var(--success)', color: '#fff'
              }}>
                Correct DPI
              </div>
              <div>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>Optimal Resolution</h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Symbols and numbers are sharp and completely visible.</p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div style={{ overflow: 'hidden', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-primary)' }}>
                  <img src={dpiCorrect1} alt="Correct DPI 1" className="w-full h-auto block transform hover:scale-105 transition-transform duration-300" />
                </div>
                <div style={{ overflow: 'hidden', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-primary)' }}>
                  <img src={dpiCorrect2} alt="Correct DPI 2" className="w-full h-auto block transform hover:scale-105 transition-transform duration-300" />
                </div>
              </div>

              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                <strong>Key Indicators:</strong> Vector lines and curves are perfectly smooth. Bounding shapes (like the circular splitter values and component boundaries) are fully closed without any cuts or truncated edges. Values can be processed accurately by the OCR model.
              </p>
            </div>

            {/* Column 2: Incorrect DPI */}
            <div style={{ 
              display: 'flex', flexDirection: 'column', gap: '1.5rem', padding: '2rem', 
              backgroundColor: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)', 
              border: '2px solid rgba(239, 68, 68, 0.4)', position: 'relative',
              boxShadow: 'var(--shadow-sm)'
            }}>
              <div style={{ 
                position: 'absolute', top: '1.25rem', right: '1.25rem', padding: '0.25rem 0.75rem', borderRadius: 'var(--radius-pill)',
                fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px',
                backgroundColor: 'var(--error)', color: '#fff'
              }}>
                Change DPI
              </div>
              <div>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>DPI Adjustment Required</h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Symbols or letters are cut off or missing borders.</p>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div style={{ overflow: 'hidden', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-primary)' }}>
                  <img src={dpiIncorrect1} alt="Incorrect DPI 1" className="w-full h-auto block transform hover:scale-105 transition-transform duration-300" />
                </div>
                <div style={{ overflow: 'hidden', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', backgroundColor: 'var(--bg-primary)' }}>
                  <img src={dpiIncorrect2} alt="Incorrect DPI 2" className="w-full h-auto block transform hover:scale-105 transition-transform duration-300" />
                </div>
              </div>

              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                <strong>Key Indicators:</strong> Bounding components (like splitter text boxes or active circles) are broken, clipped, or only partially drawn due to grid tiling boundaries. If your tile previews look like this, change the DPI setting in the prompt and re-run.
              </p>
            </div>
          </div>
        </div>
      );

      default: return <PlaceHolder name="Unknown" />;
    }
  };

  return (
    <BrowserRouter>
      <div className={`app-container ${theme}`} data-theme={theme} style={{ width: '100vw', minHeight: '100vh', overflow: 'hidden' }}>
        <AnimatePresence mode="wait">
          {isLanding ? (
            <motion.div
              key="landing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, scale: 1.05, filter: 'blur(10px)' }}
              transition={{ duration: 0.8, ease: "easeInOut" }}
            >
              <SmoothScroll>
                <LandingSection onGetStarted={() => setIsLanding(false)} />
              </SmoothScroll>
            </motion.div>
          ) : (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="layout-container"
              style={{ height: '100vh', width: '100vw', display: 'flex', overflow: 'hidden' }}
            >
              <Sidebar activeView={activeView} setActiveView={setActiveView} />

              <main className="main-content" style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <TopNav theme={theme} toggleTheme={toggleTheme} />

                <div className="page-container" ref={scrollRef} style={{ flexGrow: 1, padding: '2rem', overflowY: 'auto' }}>
                  <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={activeView}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.25, ease: "easeOut" }}
                      >
                        {renderContent()}
                      </motion.div>
                    </AnimatePresence>
                  </div>
                </div>
              </main>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </BrowserRouter>
  );
}

// ── Root export — wraps everything in SessionProvider ─────────────────────────
export default function App() {
  return (
    <SessionProvider>
      <AppInner />
    </SessionProvider>
  );
}
