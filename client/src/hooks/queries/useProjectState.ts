import { useQuery } from '@tanstack/react-query';
import { getProjectState } from '../../lib/api/projectApi';

export const useProjectState = (projectId: string | number | undefined) => {
  return useQuery({
    queryKey: ['projects', projectId, 'state'],
    queryFn: () => getProjectState(projectId!),
    enabled: !!projectId, // Only run if projectId exists
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
};
