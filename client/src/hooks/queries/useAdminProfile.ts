import { useQuery } from '@tanstack/react-query';
import { getAdminProfile } from '../../lib/api/adminApi';

export const useAdminProfile = () => {
  return useQuery({
    queryKey: ['admin', 'profile'],
    queryFn: getAdminProfile,
    staleTime: 1000 * 60 * 10, // 10 minutes - admin profile doesn't change often
  });
};
