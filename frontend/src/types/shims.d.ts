declare module 'plotly.js-cartesian-dist-min' {
  const plotly: any
  export default plotly
}

declare module 'react-plotly.js/factory' {
  import type * as React from 'react'
  const createPlotlyComponent: (plotly: any) => React.ComponentType<any>
  export default createPlotlyComponent
}
