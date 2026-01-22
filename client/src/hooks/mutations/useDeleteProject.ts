import { useMutation, useQueryClient } from '@tanstack/react-query';
import { deleteProject } from '../../lib/api/projectApi';

export const useDeleteProject = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string | number) => deleteProject(id),
    onSuccess: () => {
      // Invalidate projects list to refetch
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });
};
