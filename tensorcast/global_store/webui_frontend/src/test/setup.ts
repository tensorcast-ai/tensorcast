import '@testing-library/jest-dom'

// Mock window.location
Object.defineProperty(window, 'location', {
  value: {
    host: 'localhost:3000',
    protocol: 'http:',
  },
  writable: true,
})