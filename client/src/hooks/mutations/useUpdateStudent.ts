import { useMutation, useQueryClient } from '@tanstack/react-query';
import { updateStudent } from '../../lib/api/adminApi';
import type { UpdateStudentData } from '../../types/api';

export const useUpdateStudent = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, updates }: { id: string | number; updates: UpdateStudentData }) =>
      updateStudent(id, updates),
    onSuccess: (_data, variables) => {
      // Invalidate and refetch student detail
      queryClient.invalidateQueries({ queryKey: ['admin', 'students', variables.id] });
      // Invalidate students list to refresh the list view
      queryClient.invalidateQueries({ queryKey: ['admin', 'students'] });
    },
  });
};
