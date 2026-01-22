import { useMutation, useQueryClient } from '@tanstack/react-query';
import { saveProjectState, type ProjectState } from '../../lib/api/projectApi';

export const useSaveProject = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, state }: { id: string | number; state: ProjectState }) =>
      saveProjectState(id, state),
    onSuccess: (_data, variables) => {
      // Invalidate project state to refetch if needed
      queryClient.invalidateQueries({ queryKey: ['projects', variables.id, 'state'] });
    },
  });
};
