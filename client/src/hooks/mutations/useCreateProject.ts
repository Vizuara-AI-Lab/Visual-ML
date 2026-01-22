import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createProject, type CreateProjectData } from '../../lib/api/projectApi';

export const useCreateProject = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateProjectData) => createProject(data),
    onSuccess: () => {
      // Invalidate projects list to refetch
      queryClient.invalidateQueries({ queryKey: ['projects'] });
    },
  });
};
