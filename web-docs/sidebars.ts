import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const sidebars: SidebarsConfig = {
  // Main documentation sidebar
  docsSidebar: [
    // Introduction
    {
      type: 'doc',
      id: 'README',
      label: 'Overview',
    },
  ],

  // Developer Guides as separate sidebar
  developerGuidesSidebar: [
    {
      type: 'doc',
      id: 'developer-guides/README',
      label: 'Overview',
    },
    {
      type: 'category',
      label: 'Architecture',
      link: {
        type: 'doc',
        id: 'developer-guides/architecture/README',
      },
      items: [
        'developer-guides/architecture/architecture-overview',
        'developer-guides/architecture/global-store',
        'developer-guides/architecture/store-daemon',
        'developer-guides/architecture/high-availability-design',
        'developer-guides/architecture/p2p-transfer-strategies',
      ],
    },
    {
      type: 'category',
      label: 'Deployment',
      items: [
        'developer-guides/deployment/store-daemon',
        'developer-guides/deployment/global_store_usage',
        'developer-guides/deployment/high-availability-usage',
      ],
    },
    {
      type: 'category',
      label: 'Internals',
      items: [
        'developer-guides/internals/save_dict_flow',
        'developer-guides/internals/load_dict_pure_local_flow',
        'developer-guides/internals/model-loading',
        'developer-guides/internals/adding-metrics',
      ],
    },
    {
      type: 'category',
      label: 'AI Solutions',
      items: [
      ],
    },
    {
      type: 'category',
      label: 'Core (C++)',
      items: [
        {
          type: 'category',
          label: 'Checkpoint',
          link: {
            type: 'doc',
            id: 'developer-guides/core/checkpoint/README',
          },
          items: [
            'developer-guides/core/checkpoint/architecture',
            'developer-guides/core/checkpoint/data-format',
            'developer-guides/core/checkpoint/verification-integration',
          ],
        },
        {
          type: 'category',
          label: 'Store',
          items: [
            'developer-guides/core/store/overview',
            'developer-guides/core/store/architecture',
            'developer-guides/core/store/performance',
            'developer-guides/core/store/state-management',
          ],
        },
        {
          type: 'doc',
          id: 'developer-guides/core/communicator/README',
          label: 'Communicator',
        },
      ],
    },
  ],

};

export default sidebars;
