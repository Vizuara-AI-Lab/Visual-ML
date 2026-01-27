import { QueryClient } from "@tanstack/react-query";

// Create a client with optimized settings
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 10, // 10 minutes - data stays fresh longer
      gcTime: 1000 * 60 * 30, // 30 minutes - keep in cache longer (formerly cacheTime)
      retry: 1, // Retry failed requests once
      refetchOnWindowFocus: false, // Don't refetch on window focus
      refetchOnReconnect: false, // Don't refetch on reconnect for static data
      refetchOnMount: false, // Use cached data on mount if available
    },
    mutations: {
      retry: 0, // Don't retry mutations
    },
  },
});
