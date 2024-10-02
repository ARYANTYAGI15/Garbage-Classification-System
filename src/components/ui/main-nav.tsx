"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { motion } from "framer-motion"
import { Leaf } from "lucide-react"
import { ThemeToggle } from "./toggle-theme"

export function MainNav() {
    const pathname = usePathname()

    return (
        <div className="mr-4 flex items-center justify-between w-full">
            <div className="flex items-center space-x-6">
                <Link href="/" className="flex items-center space-x-2">
                    <motion.div
                        whileHover={{ rotate: 360 }}
                        transition={{ duration: 0.5 }}
                    >
                        <Leaf className="h-6 w-6 text-green-500" />
                    </motion.div>
                    <span className="hidden font-bold sm:inline-block text-xl">
                        EcoSort
                    </span>
                </Link>
                <nav className="flex items-center space-x-6 text-sm font-medium">
                    <Link
                        href="/"
                        className={cn(
                            "transition-colors hover:text-green-500",
                            pathname === "/" ? "text-green-500" : "text-foreground/60"
                        )}
                    >
                        Home
                    </Link>
                    <Link
                        href="/upload"
                        className={cn(
                            "transition-colors hover:text-green-500",
                            pathname?.startsWith("/upload")
                                ? "text-green-500"
                                : "text-foreground/60"
                        )}
                    >
                        Upload
                    </Link>
                </nav>
            </div>
            <ThemeToggle />
        </div>
    )
}