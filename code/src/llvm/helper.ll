; ModuleID = 'helper.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-redhat-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @_Z4testbii(i1 zeroext %x, i32 %y, i32 %z) #0 {
  %1 = alloca i8, align 1
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %q = alloca i32, align 4
  %4 = zext i1 %x to i8
  store i8 %4, i8* %1, align 1
  store i32 %y, i32* %2, align 4
  store i32 %z, i32* %3, align 4
  %5 = load i8* %1, align 1
  %6 = trunc i8 %5 to i1
  br i1 %6, label %7, label %11

; <label>:7                                       ; preds = %0
  %8 = load i32* %2, align 4
  %9 = load i32* %3, align 4
  %10 = add nsw i32 %8, %9
  store i32 %10, i32* %q, align 4
  br label %15

; <label>:11                                      ; preds = %0
  %12 = load i32* %3, align 4
  %13 = load i32* %2, align 4
  %14 = sub nsw i32 %12, %13
  store i32 %14, i32* %q, align 4
  br label %15

; <label>:15                                      ; preds = %11, %7
  %16 = load i32* %q, align 4
  ret i32 %16
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"clang version 3.4.2 (tags/RELEASE_34/dot2-final)"}
