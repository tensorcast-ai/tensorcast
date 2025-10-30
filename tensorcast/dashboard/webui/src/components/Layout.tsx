import * as React from 'react';
import { NavLink, Outlet, useMatch, useResolvedPath } from 'react-router-dom';
import { metricsConfig } from '@/lib/config';
import {
  NavigationMenu,
  NavigationMenuList,
  NavigationMenuItem,
  NavigationMenuLink,
} from '@/components/ui/navigation-menu';

type NavItemProps = {
  to: string;
  end?: boolean;
  children: React.ReactNode;
};

function NavItem({ to, end, children }: NavItemProps) {
  const resolved = useResolvedPath(to);
  const match = useMatch({ path: resolved.pathname, end: !!end });
  return (
    <NavigationMenuItem>
      <NavigationMenuLink asChild data-active={match ? true : undefined}>
        <NavLink to={to} end={end}>
          {children}
        </NavLink>
      </NavigationMenuLink>
    </NavigationMenuItem>
  );
}

export default function Layout() {
  const cfg = metricsConfig();
  return (
    <div className="min-h-screen">
      <header className="border-b">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
          <NavLink to="/" className="text-sm font-semibold hover:opacity-80">
            TensorCast Dashboard
          </NavLink>
          <NavigationMenu viewport={false}>
            <NavigationMenuList>
              <NavItem to="/" end>
                Overview
              </NavItem>
              <NavItem to="/workers">Workers</NavItem>
              <NavItem to="/replicas">Replicas</NavItem>
              {cfg.enabled ? <NavItem to="/metrics">Metrics</NavItem> : null}
            </NavigationMenuList>
          </NavigationMenu>
        </div>
      </header>
      <main className="mx-auto max-w-7xl p-4">
        <Outlet />
      </main>
    </div>
  );
}
