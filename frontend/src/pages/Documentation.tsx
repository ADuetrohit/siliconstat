import { getVersion } from '@/lib/api'
import { useAsync } from '@/lib/hooks'
import { Badge, CopyButton, NoteBanner, Panel, SectionTitle } from '@/components/ui'

const LAYERS = [
  ['circuit physics', 'level-1 MOSFET, diode, passives'],
  ['simulator', 'MNA, Newton-Raphson, AC, transient'],
  ['measurements', 'v/i/p, gain, bandwidth, offset'],
  ['mismatch model', 'global/local, Pelgrom, PVT'],
  ['Monte Carlo', 'seeding, failure accounting'],
  ['statistics', 'Welford, correlation, sensitivity'],
  ['yield', 'Wilson intervals, Cp/Cpk'],
  ['visualization', 'report, REST, this dashboard'],
  ['ML acceleration', 'surrogate trained on real runs'],
]

const NETLIST_EXAMPLE = `.title NMOS Current Mirror
.model NCH NMOS
+ VTO=0.45 KP=246u LAMBDA=0.08 GAMMA=0.40 PHI=0.80
+ AVT=3.5m ABETA=0.010

.param IB=10u
.param WM=10u

VDD  vdd 0    1.8
IREF vdd nref {IB}
M1 nref nref 0 0 NCH W={WM} L=1u MATCH=MIRROR
M2 out  nref 0 0 NCH W={WM} L=1u MATCH=MIRROR
R1 vdd out 125k

.op
.measure iref  I(M1)
.measure iout  I(M2)
.measure ierr  EXPR 100*(iout-iref)/iref
.measure ptot  P(total)
.spec ierr <= 2%
.spec ierr >= -2%
.end`

const MEASUREMENTS: [string, string, string][] = [
  ['V(node)', 'V', 'node voltage; V(a,b) for a difference'],
  ['I(device)', 'A', 'branch current, positive into the first terminal'],
  ['P(device)', 'W', 'device dissipation; P(total) is total supply power'],
  ['EXPR <expression>', '—', 'safe arithmetic over earlier measurements'],
  ['GAIN in= out= [freq=] [units=lin]', 'dB', 'AC transfer magnitude'],
  ['BW in= out=', 'Hz', 'half-power bandwidth, bisection-refined'],
  ['UGF in= out=', 'Hz', 'unity-gain frequency'],
  ['PM in= out=', 'deg', 'phase margin at the unity-gain crossing'],
  ['GM in= out=', 'dB', 'gain margin where phase reaches −180°'],
  ['VOS srcp= srcn= outp= [outn=]', 'V', 'input-referred offset, two real DC solves'],
  ['RISETIME / FALLTIME node= [lo=] [hi=]', 's', 'edge rate between two levels'],
  ['SETTLING node= [tolerance=]', 's', 'time to stay inside a band'],
  ['OVERSHOOT node=', '%', 'peak beyond the final value'],
  ['VMAX / VMIN / VPP / VAVG node=', 'V', 'transient statistics'],
]

const CORNERS: [string, string][] = [
  ['TT', 'typical NMOS, typical PMOS'],
  ['FF', 'fast NMOS, fast PMOS — lower Vth, higher β'],
  ['SS', 'slow NMOS, slow PMOS — higher Vth, lower β'],
  ['FS', 'fast NMOS, slow PMOS'],
  ['SF', 'slow NMOS, fast PMOS'],
]

function Code({ children }: { children: string }) {
  return (
    <div className="relative">
      <pre className="overflow-auto rounded-md border border-line bg-panel2 p-3 font-mono text-[11px] leading-[1.55] text-ink/90">
        {children}
      </pre>
      <div className="absolute right-1.5 top-1.5">
        <CopyButton text={children} />
      </div>
    </div>
  )
}

export default function Documentation() {
  const version = useAsync(getVersion, [])

  return (
    <div className="max-w-5xl space-y-5">
      <header>
        <h1 className="text-[18px] font-semibold text-ink">Documentation</h1>
        <p className="text-[11.5px] text-muted">
          Reference for the netlist syntax, the measurement kinds and the variation
          model. Deeper material lives in the <code>docs/</code> directory.
          {version.data && (
            <> SiliconStat v{version.data.version}.</>
          )}
        </p>
      </header>

      <Panel title="Layer hierarchy">
        <div className="space-y-1">
          {LAYERS.map(([name, detail], index) => (
            <div key={name} className="flex items-center gap-3">
              <span className="w-6 text-right text-[10px] text-muted">
                {LAYERS.length - index}
              </span>
              <span className="w-40 rounded-md border border-line bg-panel2 px-2.5 py-1 text-[11.5px] text-accent">
                {name}
              </span>
              <span className="text-[11.5px] text-muted">{detail}</span>
            </div>
          ))}
        </div>
        <p className="mt-3 text-[11px] text-muted">
          Each layer depends only on the layers beneath it. ML acceleration sits at the
          top: it is an optimisation over a pipeline that already works, trained on that
          pipeline&rsquo;s own output — never a substitute for a layer below.
        </p>
      </Panel>

      <Panel title="Netlist syntax">
        <SectionTitle hint="a complete, runnable example">Example</SectionTitle>
        <Code>{NETLIST_EXAMPLE}</Code>

        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div>
            <SectionTitle>Elements</SectionTitle>
            <ul className="space-y-1 text-[11.5px] text-muted">
              <li><code className="text-ink">R1 a b 10k</code> — resistor</li>
              <li><code className="text-ink">C1 a b 1p [IC=0.5]</code> — capacitor</li>
              <li><code className="text-ink">L1 a b 1u</code> — inductor</li>
              <li><code className="text-ink">V1 p n [DC] 1.8 [AC 1 [phase]] [PULSE(…)|SIN(…)|PWL(…)]</code></li>
              <li><code className="text-ink">I1 p n 10u</code> — current source</li>
              <li><code className="text-ink">M1 d g s b MODEL W=10u L=1u [M=2] [MATCH=GROUP]</code></li>
              <li><code className="text-ink">D1 a c DMODEL [AREA=2]</code></li>
            </ul>
          </div>
          <div>
            <SectionTitle>Directives</SectionTitle>
            <ul className="space-y-1 text-[11.5px] text-muted">
              <li><code className="text-ink">.model NAME NMOS|PMOS|D key=value …</code></li>
              <li><code className="text-ink">.param NAME=value</code>, used as <code className="text-ink">{'{NAME*2}'}</code></li>
              <li><code className="text-ink">.option reltol=1e-6 gmin=1e-12</code></li>
              <li><code className="text-ink">.temp 27</code></li>
              <li><code className="text-ink">.op</code> · <code className="text-ink">.ac dec 20 1 1G</code> · <code className="text-ink">.tran 20p 8n</code></li>
              <li><code className="text-ink">.measure NAME …</code> · <code className="text-ink">.spec NAME &lt;= 2%</code></li>
              <li><code className="text-ink">.include file.lib</code> — files only, never uploads</li>
              <li><code className="text-ink">*</code> or <code className="text-ink">;</code> comment · <code className="text-ink">+</code> continuation</li>
            </ul>
          </div>
        </div>

        <div className="mt-3">
          <NoteBanner>
            Suffixes follow SPICE: <b>M is milli, MEG is mega</b>. Expressions inside
            <code> {'{ }'}</code> accept engineering notation and are evaluated by a
            whitelisting AST walker — never <code>eval</code>.
          </NoteBanner>
        </div>
      </Panel>

      <Panel title="Measurement kinds">
        <div className="overflow-auto rounded-md border border-line">
          <table className="w-full border-collapse text-[11.5px]">
            <thead>
              <tr>
                {['Syntax', 'Unit', 'Meaning'].map((header) => (
                  <th key={header}
                      className="border-b border-line bg-panel2 px-2.5 py-1.5 text-left text-[10px] uppercase tracking-wider text-muted">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {MEASUREMENTS.map(([syntax, unit, meaning]) => (
                <tr key={syntax}>
                  <td className="border-b border-line/60 px-2.5 py-1.5 font-mono text-[11px] text-ink">
                    {syntax}
                  </td>
                  <td className="border-b border-line/60 px-2.5 py-1.5 text-muted">{unit}</td>
                  <td className="border-b border-line/60 px-2.5 py-1.5 text-muted">{meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-[11px] text-muted">
          A measurement that cannot be computed returns NaN <em>and a reason</em> — never a
          plausible substitute. Those samples are counted as
          <Badge tone="fail"> invalid_measurement</Badge> rather than as passes.
        </p>
      </Panel>

      <Panel title="Variation model">
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <SectionTitle>Global versus local</SectionTitle>
            <Code>{`Vth(M1) = Vth_nom + dVth_global(NCH) + dVth_local(M1)
Vth(M2) = Vth_nom + dVth_global(NCH) + dVth_local(M2)`}</Code>
            <p className="mt-2 text-[11.5px] text-muted">
              Two devices sharing a model card receive the <b>same</b> global draw and
              <b> independent</b> local draws. That is why a current mirror works: the
              global term cancels in the ratio and only the local terms produce a copy
              error.
            </p>
          </div>
          <div>
            <SectionTitle>Pelgrom area scaling</SectionTitle>
            <Code>{`sigma(dVT)    = AVT   / sqrt(W*L)      (matched pair)
sigma_device  = AVT   / sqrt(2*W*L)    (per device)

AVT    in V*um            e.g. AVT=3.5m
ABETA  dimensionless*um   e.g. ABETA=0.010`}</Code>
            <p className="mt-2 text-[11.5px] text-muted">
              AVT is quoted for a <b>pair</b>, so the per-device sigma is smaller by
              √2. Getting that factor wrong scales every reported sigma by 1.41.
            </p>
          </div>
        </div>
        <div className="mt-3">
          <SectionTitle>Matched groups</SectionTitle>
          <p className="text-[11.5px] text-muted">
            <code className="text-ink">MATCH=INPAIR</code> on a device puts it in a matched
            group. A global rule with <code className="text-ink">share_by=&quot;matched_group&quot;</code>
            {' '}then shares one draw within the group rather than across the whole model
            card — the way to express &ldquo;these devices are laid out together&rdquo;.
          </p>
        </div>
      </Panel>

      <Panel title="PVT corners">
        <div className="overflow-auto rounded-md border border-line">
          <table className="w-full border-collapse text-[11.5px]">
            <tbody>
              {CORNERS.map(([name, meaning]) => (
                <tr key={name}>
                  <td className="w-16 border-b border-line/60 px-2.5 py-1.5 text-accent">
                    {name}
                  </td>
                  <td className="border-b border-line/60 px-2.5 py-1.5 text-muted">
                    {meaning}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-[11px] text-muted">
          Corners are defined as ±3σ of the <em>same</em> global distribution the Monte
          Carlo process model uses, which keeps the two views consistent. That is not how
          a foundry defines corner models, and the difference is documented in
          <code> docs/limitations.md</code>.
        </p>
      </Panel>

      <Panel title="How yield is counted">
        <ul className="space-y-1.5 text-[11.5px] text-muted">
          <li>
            <b className="text-ink">Two denominators, always both.</b> Over successful
            simulations (what a designer wants) and over all attempted samples (the
            conservative bound). The gap is what the run lost to numerical failure.
          </li>
          <li>
            <b className="text-ink">Combined yield is counted sample by sample</b>, not
            multiplied. Specifications that share a variation source fail together, so
            the product overstates the true yield.
          </li>
          <li>
            <b className="text-ink">Wilson score intervals</b>, because the normal
            approximation claims a 50/50 run proves 100 % yield.
          </li>
          <li>
            <b className="text-ink">An invalid measurement fails its specs</b> and is
            reported in a note. A NaN is not a pass.
          </li>
        </ul>
      </Panel>

      <Panel title="Limitations">
        <div className="space-y-2">
          <NoteBanner>
            The MOSFET model is square-law (SPICE level-1 class): no velocity saturation,
            no DIBL, no subthreshold conduction. It is <b>not BSIM</b>, it is not
            calibrated to silicon, and the shipped model cards are plausible 180 nm-class
            numbers rather than a PDK.
          </NoteBanner>
          <NoteBanner>
            Transient analysis is fixed-step with no local-truncation-error control; the
            matrix solver is dense; there are no subcircuits, no noise analysis and no
            aging model.
          </NoteBanner>
          <NoteBanner>
            Plain Monte Carlo cannot resolve ppm yields — that needs importance sampling,
            which is not implemented. The Yield page will tell you how many samples a
            target precision actually requires.
          </NoteBanner>
          <NoteBanner>
            <b>This is a tool for education, methodology demonstration and design
            exploration. It is not a sign-off tool.</b>
          </NoteBanner>
        </div>
      </Panel>

      <Panel title="Further reading">
        <ul className="grid gap-1 text-[11.5px] text-muted md:grid-cols-2">
          {[
            ['docs/architecture.md', 'layers, data flow, security model'],
            ['docs/circuit_solver.md', 'MNA, Newton-Raphson, AC, transient'],
            ['docs/mosfet_model.md', 'the equations and what is not modelled'],
            ['docs/monte_carlo.md', 'seeding, failures, parallelism, throughput'],
            ['docs/mismatch.md', 'global/local, copulas, correlation'],
            ['docs/pelgrom.md', 'area scaling, measured, plus the trap it invites'],
            ['docs/statistics.md', 'Welford, percentiles, sensitivity methods'],
            ['docs/yield.md', 'denominators, Wilson, Cp/Cpk, sample planning'],
            ['docs/validation.md', 'every closed-form check, with numbers'],
            ['docs/limitations.md', 'the unflattering list'],
            ['docs/api_contract.md', 'the REST contract this UI consumes'],
          ].map(([file, blurb]) => (
            <li key={file}>
              <code className="text-ink">{file}</code>
              <span className="text-muted"> — {blurb}</span>
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  )
}
