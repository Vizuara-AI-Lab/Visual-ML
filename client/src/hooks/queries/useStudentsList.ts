import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { getStudentsList } from '../../lib/api/adminApi';
import type { StudentFilters } from '../../types/api';

export const useStudentsList = (filters: StudentFilters = {}) => {
  return useQuery({
    queryKey: ['admin', 'students', filters],
    queryFn: () => getStudentsList(filters),
    staleTime: 1000 * 60 * 2, // 2 minutes
    placeholderData: keepPreviousData,
  });
};
