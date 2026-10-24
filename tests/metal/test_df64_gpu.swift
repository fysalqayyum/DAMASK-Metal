// GPU validation of df64.h on the Metal device: compares against CPU double.
// run: swift tests/metal/test_df64_gpu.swift src/grid/metal/df64.h
import Metal
import Foundation

let header = try! String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
let src = header + """

kernel void k(device const float2* a [[buffer(0)]], device const float2* b [[buffer(1)]],
              device float2* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  df64 A = df_make(a[i].x, a[i].y), B = df_make(b[i].x, b[i].y);
  df64 r[6] = { A*B, A/B, df_sqrt(df_abs(A)), df_exp(B), df_log(df_abs(A)), df_pow(df_abs(B)+0.05f, df_from(20.0f)) };
  for (int j = 0; j < 6; ++j) out[6*i+j] = float2(r[j].hi, r[j].lo);
}
"""
let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions()
opts.mathMode = .safe
let lib = try! dev.makeLibrary(source: src, options: opts)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "k")!)

let n = 1 << 20
func split(_ x: Double) -> SIMD2<Float> { let h = Float(x); return SIMD2(h, Float(x - Double(h))) }
var a = [SIMD2<Float>](), b = [SIMD2<Float>](), ad = [Double](), bd = [Double]()
var g = SystemRandomNumberGenerator()
for _ in 0..<n {
  let x = split(Double.random(in: -1e3...1e3, using: &g)); a.append(x); ad.append(Double(x.x)+Double(x.y))
  let y = split(Double.random(in: -2...2, using: &g)); b.append(y); bd.append(Double(y.x)+Double(y.y))
}
let A = dev.makeBuffer(bytes: a, length: n*8, options: .storageModeShared)!
let B = dev.makeBuffer(bytes: b, length: n*8, options: .storageModeShared)!
let O = dev.makeBuffer(length: n*48, options: .storageModeShared)!
let q = dev.makeCommandQueue()!, cb = q.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pso); enc.setBuffer(A, offset: 0, index: 0); enc.setBuffer(B, offset: 0, index: 1); enc.setBuffer(O, offset: 0, index: 2)
enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
let o = O.contents().bindMemory(to: SIMD2<Float>.self, capacity: 6*n)
var err = [Double](repeating: 0, count: 6)
let names = ["mul", "div", "sqrt", "exp", "log", "pow(x,20)"]
for i in 0..<n {
  let ref = [ad[i]*bd[i], ad[i]/bd[i], abs(ad[i]).squareRoot(), exp(bd[i]), log(abs(ad[i])), pow(abs(bd[i])+Double(Float(0.05)), 20)]
  for j in 0..<6 {
    let v = Double(o[6*i+j].x) + Double(o[6*i+j].y)
    err[j] = max(err[j], abs(v-ref[j]) / max(abs(ref[j]), 1e-300) / (j == 4 ? max(1, abs(ref[j]))/abs(ref[j]) : 1))
  }
}
print("GPU:", dev.name, " samples:", n)
for j in 0..<6 { print(String(format: "  %-10@ max rel err %.2e", names[j] as NSString, err[j])) }
exit(err.allSatisfy { $0 < 1e-11 } ? 0 : 1)
