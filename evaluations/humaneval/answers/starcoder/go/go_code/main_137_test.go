package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestCompareOne(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(2, CompareOne(1, 2))
    assert.Equal(2.5, CompareOne(1, 2.5))
    assert.Equal(3, CompareOne(2, 3))
    assert.Equal(6, CompareOne(5, 6))
    assert.Equal("2,3", CompareOne(1, "2,3"))
    assert.Equal("6", CompareOne("5,1", "6"))
    assert.Equal("2", CompareOne("1", "2"))
    assert.Equal(nil, CompareOne("1", 1))
}
