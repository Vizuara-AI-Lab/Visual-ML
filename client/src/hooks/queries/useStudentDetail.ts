import { useQuery } from '@tanstack/react-query';
import { getStudentById } from '../../lib/api/adminApi';

export const useStudentDetail = (id: string | number | undefined) => {
  return useQuery({
    queryKey: ['admin', 'students', id],
    queryFn: () => getStudentById(id!),
    enabled: !!id, // Only run query if id exists
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
};
