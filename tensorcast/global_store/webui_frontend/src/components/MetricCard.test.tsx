import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import MetricCard from './MetricCard'

describe('MetricCard', () => {
  it('renders title and value', () => {
    render(<MetricCard title="Test Metric" value={42} />)

    expect(screen.getByText('Test Metric')).toBeInTheDocument()
    expect(screen.getByText('42')).toBeInTheDocument()
  })

  it('renders suffix when provided', () => {
    render(<MetricCard title="Test" value={10} suffix="/ 100" />)

    expect(screen.getByText('/ 100')).toBeInTheDocument()
  })

  it('shows loading state', () => {
    const { container } = render(<MetricCard title="Test" value={0} loading />)

    expect(container.querySelector('.ant-skeleton')).toBeInTheDocument()
  })
})