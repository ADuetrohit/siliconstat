import { useCallback, useEffect, useState } from 'react'

export interface AsyncState<T> {
  data: T | null
  error: string | null
  loading: boolean
  reload: () => void
}

/** Run an async loader on mount and whenever `deps` change. */
export function useAsync<T>(loader: () => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    let alive = true
    setLoading(true)
    setError(null)
    loader()
      .then((value) => { if (alive) { setData(value); setLoading(false) } })
      .catch((err: Error) => {
        if (alive) { setError(err.message); setData(null); setLoading(false) }
      })
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])
  return { data, error, loading, reload }
}

/** Persist a value in localStorage so a page selection survives navigation. */
export function useStickyState<T>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = window.localStorage.getItem(key)
      return stored === null ? initial : (JSON.parse(stored) as T)
    } catch {
      return initial
    }
  })
  const update = useCallback((next: T) => {
    setValue(next)
    try {
      window.localStorage.setItem(key, JSON.stringify(next))
    } catch {
      /* storage unavailable; in-memory is fine */
    }
  }, [key])
  return [value, update]
}
