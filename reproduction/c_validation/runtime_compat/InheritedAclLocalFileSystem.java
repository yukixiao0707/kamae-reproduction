// Copyright [2024] Expedia, Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS.
package cvalidation;

import java.io.IOException;
import java.nio.file.Paths;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RawLocalFileSystem;
import org.apache.hadoop.fs.permission.FsPermission;

/** 仅在实验目录保留继承的 Windows ACL；文件内容读写仍使用 Hadoop 原实现。 */
public final class InheritedAclLocalFileSystem extends LocalFileSystem {
    public InheritedAclLocalFileSystem() {
        super(new InheritedAclRawLocalFileSystem());
    }

    public static final class InheritedAclRawLocalFileSystem extends RawLocalFileSystem {
        @Override
        public void setPermission(Path path, FsPermission permission) throws IOException {
            String text = path.toUri().getPath();
            if (text.matches("^/[A-Za-z]:/.*")) {
                text = text.substring(1);
            }
            java.nio.file.Path absolute = Paths.get(text).toAbsolutePath().normalize();
            for (String name : new String[] {"KAMAE_C_REPO_ROOT", "KAMAE_C_TEMP_ROOT"}) {
                String root = System.getenv(name);
                if (root != null && absolute.startsWith(Paths.get(root).toAbsolutePath().normalize())) {
                    // 不扩大权限，也不修改 ACL；仅保留操作系统创建文件时继承的权限。
                    return;
                }
            }
            super.setPermission(path, permission);
        }
    }
}
