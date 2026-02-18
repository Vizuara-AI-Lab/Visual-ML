import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  FolderKanban,
  FileText,
  AlertCircle,
  Loader2,
  Sparkles,
} from "lucide-react";
import { useCreateProject } from "../../hooks/mutations/useCreateProject";

interface CreateProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (projectId: number) => void;
}

export const CreateProjectModal: React.FC<CreateProjectModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const createProject = useCreateProject();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    try {
      const project = await createProject.mutateAsync({
        name: name.trim(),
        description: description.trim() || undefined,
      });
      setName("");
      setDescription("");
      onSuccess(project.id);
    } catch (error) {
      console.error("Failed to create project:", error);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <motion.div
            className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            className="relative bg-white/80 backdrop-blur-xl rounded-2xl shadow-2xl shadow-slate-900/20 max-w-md w-full mx-4 border border-slate-200/60 ring-1 ring-slate-900/5 overflow-hidden"
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
          >
            {/* Top accent line */}
            <div className="absolute top-0 left-0 right-0 h-1 bg-linear-to-r from-slate-900 via-slate-700 to-slate-900" />

            {/* Header */}
            <div className="flex items-center justify-between p-6 pb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-linear-to-br from-slate-900 to-slate-700 flex items-center justify-center shadow-lg shadow-slate-900/25">
                  <FolderKanban className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold bg-linear-to-br from-slate-900 via-slate-800 to-slate-900 bg-clip-text text-transparent">
                    Create New Project
                  </h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Start building your ML pipeline
                  </p>
                </div>
              </div>
              <motion.button
                onClick={onClose}
                className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100/80 rounded-xl transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <X className="h-5 w-5" />
              </motion.button>
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="px-6 pb-6">
              <div className="space-y-4">
                {/* Project Name */}
                <div>
                  <label
                    htmlFor="modal-name"
                    className="block text-sm font-semibold text-slate-700 mb-2"
                  >
                    Project Name <span className="text-red-500">*</span>
                  </label>
                  <div className="relative">
                    <div className="absolute left-4 top-1/2 -translate-y-1/2">
                      <Sparkles className="h-4 w-4 text-slate-400" />
                    </div>
                    <input
                      type="text"
                      id="modal-name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="My ML Project"
                      className="w-full pl-11 pr-4 py-3 bg-white/60 border border-slate-200/60 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-900 focus:border-transparent transition-all shadow-sm hover:shadow-md backdrop-blur-sm font-medium text-slate-900 placeholder:text-slate-400"
                      required
                      maxLength={255}
                    />
                  </div>
                </div>

                {/* Description */}
                <div>
                  <label
                    htmlFor="modal-description"
                    className="block text-sm font-semibold text-slate-700 mb-2"
                  >
                    Description{" "}
                    <span className="text-slate-400 font-normal">
                      (Optional)
                    </span>
                  </label>
                  <div className="relative">
                    <div className="absolute left-4 top-3.5">
                      <FileText className="h-4 w-4 text-slate-400" />
                    </div>
                    <textarea
                      id="modal-description"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="Describe your project..."
                      rows={3}
                      className="w-full pl-11 pr-4 py-3 bg-white/60 border border-slate-200/60 rounded-xl focus:outline-none focus:ring-2 focus:ring-slate-900 focus:border-transparent transition-all shadow-sm hover:shadow-md backdrop-blur-sm text-slate-900 placeholder:text-slate-400 resize-none"
                      maxLength={1000}
                    />
                  </div>
                </div>
              </div>

              {/* Error */}
              <AnimatePresence>
                {createProject.isError && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="mt-4 p-3 bg-red-50 border border-red-200 rounded-xl flex items-start gap-2"
                  >
                    <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                    <p className="text-sm text-red-700">
                      {createProject.error instanceof Error
                        ? createProject.error.message
                        : "Failed to create project"}
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Actions */}
              <div className="flex items-center justify-end gap-3 mt-6">
                <motion.button
                  type="button"
                  onClick={onClose}
                  className="px-5 py-2.5 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100/80 rounded-xl transition-colors"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Cancel
                </motion.button>
                <motion.button
                  type="submit"
                  disabled={!name.trim() || createProject.isPending}
                  className="px-5 py-2.5 text-sm font-semibold text-white bg-slate-900 hover:bg-slate-800 rounded-xl transition-all shadow-lg shadow-slate-900/25 hover:shadow-xl hover:shadow-slate-900/40 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  whileHover={
                    !name.trim() || createProject.isPending
                      ? {}
                      : { scale: 1.02, y: -1 }
                  }
                  whileTap={
                    !name.trim() || createProject.isPending
                      ? {}
                      : { scale: 0.98 }
                  }
                >
                  {createProject.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Creating...</span>
                    </>
                  ) : (
                    <>
                      <FolderKanban className="w-4 h-4" />
                      <span>Create Project</span>
                    </>
                  )}
                </motion.button>
              </div>
            </form>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
