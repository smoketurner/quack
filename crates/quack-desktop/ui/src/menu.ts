// What the native menu asked for, handed to whichever view acts on it.
// Views subscribe in an effect and update their state in the callback, as
// for any external source of events.
export type MenuHandler = (id: string) => void;

export interface MenuBus {
  subscribe(handler: MenuHandler): () => void;
  emit(id: string): void;
}

export function menuBus(): MenuBus {
  const handlers = new Set<MenuHandler>();
  return {
    subscribe(handler) {
      handlers.add(handler);
      return () => {
        handlers.delete(handler);
      };
    },
    emit(id) {
      for (const handler of handlers) handler(id);
    },
  };
}
