import { useQuery } from "@tanstack/react-query";
import { getProjects } from "../../lib/api/projectApi";

export const useProjects = () => {
  return useQuery({
    queryKey: ["projects"],
    queryFn: getProjects,
    staleTime: 1000 * 60 * 15, // 15 minutes - projects don't change often
  });
};
